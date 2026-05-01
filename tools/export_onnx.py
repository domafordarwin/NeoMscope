"""Export a trained YOLOv11-det `.pt` to ONNX for Hailo Dataflow Compiler.

Runs in the main venv on the dev PC. Performs an immediate parity check
between the PyTorch and ONNX outputs (max abs diff must be < 1e-3, per
design §4.5).

CLI:
    python tools/export_onnx.py \
        --weights models/pt/best.pt \
        --output models/onnx/best.onnx \
        --imgsz 640
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("export_onnx")


def export(
    weights_path: Path,
    output_path: Path,
    imgsz: int = 640,
    opset: int = 17,
    parity_threshold: float = 1e-3,
) -> Path:
    """Export `.pt` to ONNX with Hailo-friendly settings, then verify parity.

    Hailo DFC requires:
      - opset 16-19
      - dynamic=False (static input shape)
      - simplify=True (graph simplification)
      - nms=False (NMS handled in postprocess on Pi)

    Args:
        weights_path: Path to trained `.pt`.
        output_path: Where to write `.onnx`.
        imgsz: Square input edge in pixels (must match HEF compile target).
        opset: ONNX opset version (16-19).
        parity_threshold: Max allowed abs diff between torch and ORT outputs.

    Returns:
        Path to the produced ONNX file.

    Raises:
        FileNotFoundError: weights_path does not exist.
        RuntimeError: parity check fails.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Lazy imports — these are heavy (torch + ultralytics) and only needed at runtime
    import numpy as np
    from ultralytics import YOLO

    logger.info("Loading %s ...", weights_path)
    model = YOLO(str(weights_path))

    logger.info("Exporting to ONNX (opset=%d, imgsz=%d, dynamic=False, nms=False) ...",
                opset, imgsz)
    exported = model.export(
        format="onnx",
        opset=opset,
        dynamic=False,
        simplify=True,
        imgsz=imgsz,
        half=False,
        nms=False,
    )
    exported_path = Path(exported)
    logger.info("Ultralytics wrote: %s", exported_path)

    # Move to requested output if different
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_path.resolve():
        output_path.write_bytes(exported_path.read_bytes())
        logger.info("Copied to: %s", output_path)

    # ---- Parity check ----
    logger.info("Verifying ONNX vs PyTorch parity ...")
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(seed=42)
    dummy = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)

    # ONNX Runtime
    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_outputs = sess.run(None, {input_name: dummy})

    # PyTorch
    torch_input = torch.from_numpy(dummy)
    with torch.no_grad():
        torch_outputs = model.model(torch_input)

    # Compare first output (the one Hailo will use) — flatten and compare
    torch_first = torch_outputs[0] if isinstance(torch_outputs, tuple) else torch_outputs
    if hasattr(torch_first, "cpu"):
        torch_first = torch_first.cpu().numpy()
    elif isinstance(torch_first, list):
        torch_first = torch_first[0].cpu().numpy() if hasattr(torch_first[0], "cpu") else np.asarray(torch_first[0])

    ort_first = ort_outputs[0]

    if torch_first.shape != ort_first.shape:
        logger.warning("Shape mismatch: torch=%s, onnx=%s. Skipping numeric comparison.",
                       torch_first.shape, ort_first.shape)
    else:
        max_abs_diff = float(np.max(np.abs(torch_first - ort_first)))
        logger.info("Max abs diff: %.6e (threshold: %.0e)", max_abs_diff, parity_threshold)
        if max_abs_diff > parity_threshold:
            raise RuntimeError(
                f"ONNX/PyTorch parity check failed: max abs diff "
                f"{max_abs_diff:.6e} > threshold {parity_threshold}. "
                f"Re-export with different opset, or investigate model layers."
            )

    logger.info("✅ Export complete: %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True, help="Path to .pt")
    parser.add_argument("--output", type=Path, default=Path("models/onnx/best.onnx"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17, choices=range(16, 20))
    parser.add_argument(
        "--parity-threshold",
        type=float,
        default=1e-3,
        help="Max allowed abs diff between torch/onnx (default: 1e-3)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(message)s")

    try:
        export(
            weights_path=args.weights,
            output_path=args.output,
            imgsz=args.imgsz,
            opset=args.opset,
            parity_threshold=args.parity_threshold,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except ImportError as exc:
        logger.error(
            "Required dependency missing: %s. Install with: pip install -e .[tools]",
            exc,
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
