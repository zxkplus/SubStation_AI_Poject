"""Tests for convert category filtering in dataset_cropper.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dataset_cropper import DatasetCropper


def test_only_categories_filters_recursive(tmp_path):
    root = tmp_path / "input"
    out = tmp_path / "out"
    for category in ("散热器", "隔离开关"):
        category_dir = root / category
        category_dir.mkdir(parents=True)
        (category_dir / "a.jpg").write_bytes(b"x")
        (category_dir / "a.json").write_text(
            json.dumps({"shapes": [], "imageWidth": 10, "imageHeight": 10}),
            encoding="utf-8",
        )

    cropper = DatasetCropper(str(root), str(out), only_categories=["散热器"])
    structure = cropper.load_dataset_structure(recursive=True)

    assert list(structure) == ["散热器"]
