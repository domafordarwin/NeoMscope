"""Hailo NPU inference pipeline — Pi 5 + AI HAT+ 2 (Hailo-10H).

Loads a compiled `.hef`, accepts BGR frames, returns `Detection` lists.
The Hailo runtime imports are deferred so that this module can be
imported on the dev PC for type checking and signature inspection.

The default code path uses **HailoRT Python directly** (no GStreamer).
GStreamer + `hailo_apps_infra` would be lower-latency for video streams
but requires a build toolchain; we keep that path optional behind
`PipelineConfig.use_gstreamer=True` and lazy-import.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from inference.postprocess import decode_yolo_det_output
from inference.types import Detection, PipelineConfig, letterbox_size

if TYPE_CHECKING:  # avoid hard dependency on dev PC
    pass

logger = logging.getLogger("pipeline")


def _letterbox(frame: np.ndarray, target: int) -> np.ndarray:
    """Resize frame to (target, target) preserving aspect ratio with padding.

    Returns BGR uint8 of shape (target, target, 3). Padding is gray (114).
    Inverse handled by `inference.types.unletterbox_bbox` during postprocess.
    """
    h, w = frame.shape[:2]
    new_h, new_w, _scale = letterbox_size((h, w), target)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    out = np.full((target, target, 3), 114, dtype=np.uint8)
    out[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return out


class HailoInferencePipeline:
    """Stateful wrapper around a HailoRT VDevice + InferenceModel.

    Lifecycle:
        with HailoInferencePipeline(cfg) as pipe:
            for frame in source.frames():
                detections = pipe.infer(frame)

    Implementation detail: we lazy-import `hailo_platform` inside `__init__`
    so unit tests on the dev PC can construct the class with `mock=True`
    and exercise the surrounding logic.
    """

    def __init__(self, config: PipelineConfig, *, mock: bool = False) -> None:
        self._config = config
        self._mock = mock
        self._infer_model = None
        self._configured_infer_model = None
        self._vdevice = None
        self._input_name: str | None = None
        self._output_names: list[str] = []

        if mock:
            logger.warning("HailoInferencePipeline running in MOCK mode (no NPU calls).")
            return

        self._open_hailo()

    def _open_hailo(self) -> None:
        if not self._config.hef_path.is_file():
            raise FileNotFoundError(f"HEF not found: {self._config.hef_path}")

        # Defer import — only available on Pi with HailoRT installed
        try:
            from hailo_platform import (  # type: ignore[import-not-found]
                HEF,
                ConfigureParams,
                VDevice,
            )
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform not installed. Run setup.sh on the Pi 5 to install HailoRT."
            ) from exc

        hef = HEF(str(self._config.hef_path))
        self._vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=self._vdevice.interface_type
        )
        network_groups = self._vdevice.configure(hef, configure_params)
        self._infer_model = network_groups[0]

        # Cache input/output bindings
        input_vstream_infos = hef.get_input_vstream_infos()
        output_vstream_infos = hef.get_output_vstream_infos()
        if len(input_vstream_infos) != 1:
            raise RuntimeError(
                f"Expected single HEF input, got {len(input_vstream_infos)}"
            )
        self._input_name = input_vstream_infos[0].name
        self._output_names = [info.name for info in output_vstream_infos]
        logger.info(
            "HEF loaded: 1 input (%s), %d outputs (%s)",
            self._input_name,
            len(self._output_names),
            self._output_names,
        )

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Run end-to-end inference: preprocess → NPU → postprocess.

        Args:
            frame: BGR uint8 image of shape (H, W, 3) at any resolution.

        Returns:
            List of Detection records in original-frame coordinates.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) BGR, got {frame.shape}")

        orig_h, orig_w = frame.shape[:2]
        letterboxed = _letterbox(frame, self._config.image_size)

        # Hailo expects RGB float32 in [0, 1] (or uint8 depending on HEF —
        # the Hailo Model Zoo YOLO config uses uint8 input with on-NPU
        # normalization, so we send uint8 RGB).
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        nchw_input = np.expand_dims(rgb, axis=0)  # (1, H, W, 3) NHWC

        if self._mock:
            # Synthetic outputs for unit tests: single high-conf detection
            n_anchors = 8400
            bbox = np.zeros((n_anchors, 4), dtype=np.float32)
            cls = np.zeros((n_anchors, 5), dtype=np.float32)
            bbox[0] = [0.5, 0.5, 0.4, 0.4]
            cls[0, 0] = 0.95
            raw_outputs = {"bbox": bbox, "cls": cls}
        else:
            raw_outputs = self._run_hailo_inference(nchw_input)

        return decode_yolo_det_output(
            raw_outputs,
            orig_shape=(orig_h, orig_w),
            image_size=self._config.image_size,
            conf_threshold=self._config.conf_threshold,
            iou_threshold=self._config.iou_threshold,
            max_detections=self._config.max_detections,
        )

    def _run_hailo_inference(self, nhwc_input: np.ndarray) -> dict[str, np.ndarray]:
        """Submit one frame to the NPU and unpack outputs.

        The exact API surface of `hailo_platform` shifts between versions;
        this implementation targets HailoRT 4.20+. If your installed
        version differs, update the `with` block accordingly.
        """
        from hailo_platform import (  # type: ignore[import-not-found]
            FormatType,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
        )

        input_params = InputVStreamParams.make(self._infer_model, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(
            self._infer_model, format_type=FormatType.FLOAT32
        )

        with (
            InferVStreams(self._infer_model, input_params, output_params) as infer_pipeline,
            self._infer_model.activate(),
        ):
            results = infer_pipeline.infer({self._input_name: nhwc_input})

        # Map Hailo output names to the canonical 'bbox' and 'cls' keys
        # expected by postprocess.decode_yolo_det_output. Concrete mapping
        # depends on the YOLO11 head names emitted by the Model Zoo
        # config — adjust here once the HEF is compiled and inspected.
        return {
            "bbox": results[self._output_names[0]][0],   # (N, 4)
            "cls": results[self._output_names[1]][0],    # (N, NUM_CLASSES)
        }

    def close(self) -> None:
        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception as exc:
                # Cleanup is best-effort — log and move on regardless of the error.
                logger.warning("VDevice.release raised: %s", exc)
            self._vdevice = None

    def __enter__(self) -> HailoInferencePipeline:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
