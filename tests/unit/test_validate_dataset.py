"""Unit tests for tools/validate_dataset.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import validate_dataset as vd


class TestValidateLine:
    @pytest.mark.parametrize(
        "line",
        [
            "0 0.5 0.5 0.2 0.2",
            "4 0.0 0.0 1.0 1.0",
            "2 0.123456 0.789012 0.05 0.05",
        ],
    )
    def test_valid_lines(self, line: str) -> None:
        ok, err = vd._validate_line(line)
        assert ok, f"expected valid but got error: {err}"

    @pytest.mark.parametrize(
        ("line", "reason"),
        [
            ("", "expected 5 fields"),
            ("0 0.5 0.5 0.2", "expected 5 fields"),
            ("0 0.5 0.5 0.2 0.2 extra", "expected 5 fields"),
            ("five 0.5 0.5 0.2 0.2", "could not parse"),
            ("-1 0.5 0.5 0.2 0.2", "out of range"),
            ("5 0.5 0.5 0.2 0.2", "out of range"),
            ("0 1.1 0.5 0.2 0.2", "not in"),
            ("0 0.5 -0.1 0.2 0.2", "not in"),
            ("0 0.5 0.5 0.0 0.2", "degenerate"),
            ("0 0.5 0.5 0.2 0.0", "degenerate"),
        ],
    )
    def test_invalid_lines(self, line: str, reason: str) -> None:
        ok, err = vd._validate_line(line)
        assert not ok
        assert reason in err


class TestValidate:
    def _make_dataset(
        self,
        root: Path,
        items: dict[str, str],
    ) -> Path:
        """Build a tiny dataset with given {stem: label_text} pairs.

        Empty label_text means image with no annotations.
        Stems prefixed with '!' have no .jpg (only .txt) → labels_without_images.
        Stems suffixed with '*' have no .txt (only .jpg) → images_without_labels.
        """
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        for stem, text in items.items():
            has_image = not stem.startswith("!")
            has_label = not stem.endswith("*")
            clean_stem = stem.lstrip("!").rstrip("*")
            if has_image:
                (root / "images" / f"{clean_stem}.jpg").touch()
            if has_label:
                (root / "labels" / f"{clean_stem}.txt").write_text(text)
        return root

    def test_passes_on_clean_dataset(self, tmp_path: Path) -> None:
        root = self._make_dataset(
            tmp_path / "ds",
            {
                "img01": "0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1",
                "img02": "1 0.7 0.7 0.15 0.15",
                "img03": "",  # empty label is valid
            },
        )
        report = vd.validate(root)
        assert report.passed
        assert report.total_images == 3
        assert report.total_labels == 3
        assert report.total_annotations == 3
        assert report.class_counts[0] == 1
        assert report.class_counts[1] == 1
        assert report.class_counts[2] == 1
        assert len(report.empty_label_files) == 1

    def test_detects_image_without_label(self, tmp_path: Path) -> None:
        root = self._make_dataset(
            tmp_path / "ds",
            {
                "good": "0 0.5 0.5 0.2 0.2",
                "missing*": "0 0.5 0.5 0.2 0.2",  # ! removed image, label stays
            },
        )
        # Actually our marker logic: '*' suffix means no .txt. Reverse: prefix '!' means no image.
        report = vd.validate(root)
        # 'missing' (with * suffix removed) has image but no label → in images_without_labels
        assert "missing.jpg" in report.images_without_labels

    def test_detects_invalid_class(self, tmp_path: Path) -> None:
        root = self._make_dataset(
            tmp_path / "ds",
            {"bad": "9 0.5 0.5 0.2 0.2"},
        )
        report = vd.validate(root)
        assert not report.passed
        assert any("out of range" in err for _, _, err in report.invalid_lines)

    def test_class_imbalance_calculation(self, tmp_path: Path) -> None:
        # Imbalanced: 5 of class 0, 1 of class 1
        text = "\n".join(["0 0.5 0.5 0.1 0.1"] * 5 + ["1 0.5 0.5 0.1 0.1"])
        root = self._make_dataset(tmp_path / "ds", {"img": text})
        report = vd.validate(root)
        assert report.class_imbalance() == 5.0

    def test_raises_on_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            vd.validate(tmp_path / "nope")

    def test_raises_on_missing_subdirs(self, tmp_path: Path) -> None:
        (tmp_path / "ds").mkdir()
        with pytest.raises(FileNotFoundError):
            vd.validate(tmp_path / "ds")
