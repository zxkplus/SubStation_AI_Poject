"""Tests for --remap_ignore_to behavior in train_yolo.py."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_yolo import (
    filter_label_file,
    generate_data_yaml,
    prepare_dataset,
    remap_label_class_ids,
)


def write_classes(dataset_dir, class_names):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    classes_file = dataset_dir / "classes.txt"
    classes_file.write_text(
        "\n".join(f"{i} {name}" for i, name in enumerate(class_names)) + "\n",
        encoding="utf-8",
    )


def write_label(label_path, lines):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_generate_data_yaml_remaps_ignored_ids(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_classes(dataset, ["bei_jing", "you_zhen", "she_bei_biao_shi", "dian_kang_qi"])

    output_yaml = tmp_path / "out" / "data.yaml"
    _, old_id_to_new_id = generate_data_yaml(
        str(dataset),
        str(output_yaml),
        train_ratio=0.8,
        val_ratio=0.2,
        ignore_classes=["you_zhen", "dian_kang_qi"],
        remap_target_class="bei_jing",
    )

    config = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert config["nc"] == 2
    assert config["names"] == {0: "bei_jing", 1: "she_bei_biao_shi"}
    assert old_id_to_new_id == {0: 0, 1: 0, 2: 1, 3: 0}


def test_filter_label_file_remaps_ignored_classes(tmp_path):
    label_path = tmp_path / "labels" / "sample.txt"
    write_label(label_path, [
        "0 0.1 0.2 0.3 0.4",
        "1 0.2 0.3 0.4 0.5",
        "2 0.3 0.4 0.5 0.6",
        "3 0.4 0.5 0.6 0.7",
    ])
    class_mapping = {
        "bei_jing": 0,
        "you_zhen": 1,
        "she_bei_biao_shi": 2,
        "dian_kang_qi": 3,
    }

    valid = filter_label_file(
        label_path,
        class_mapping,
        ["you_zhen", "dian_kang_qi"],
        remap_target_class="bei_jing",
    )

    assert valid
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    class_ids = [line.split()[0] for line in lines]
    assert class_ids == ["0", "0", "2", "0"]


def test_filter_label_file_without_remap_still_deletes_ignored_only(tmp_path):
    label_path = tmp_path / "labels" / "sample.txt"
    write_label(label_path, ["1 0.2 0.3 0.4 0.5"])
    class_mapping = {"bei_jing": 0, "you_zhen": 1}

    valid = filter_label_file(label_path, class_mapping, ["you_zhen"])

    assert not valid
    assert not label_path.exists()


def test_prepare_dataset_keeps_ignored_only_image_when_remapping(tmp_path):
    dataset = tmp_path / "yolo_dataset"
    write_classes(dataset, ["bei_jing", "you_zhen", "she_bei_biao_shi"])

    train_img_dir = dataset / "train" / "images"
    train_lbl_dir = dataset / "train" / "labels"
    val_img_dir = dataset / "val" / "images"
    val_lbl_dir = dataset / "val" / "labels"
    for directory in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    (train_img_dir / "train.jpg").touch()
    (val_img_dir / "val.jpg").touch()
    write_label(train_lbl_dir / "train.txt", ["1 0.2 0.3 0.4 0.5"])
    write_label(val_lbl_dir / "val.txt", ["1 0.6 0.7 0.8 0.9"])

    class_mapping = {
        "bei_jing": 0,
        "you_zhen": 1,
        "she_bei_biao_shi": 2,
    }
    prepare_dataset(
        str(dataset),
        train_ratio=0.8,
        val_ratio=0.2,
        ignore_classes=["you_zhen"],
        remap_target_class="bei_jing",
    )

    output_yaml = tmp_path / "out" / "data.yaml"
    _, old_id_to_new_id = generate_data_yaml(
        str(dataset),
        str(output_yaml),
        ignore_classes=["you_zhen"],
        remap_target_class="bei_jing",
    )
    remap_label_class_ids(train_lbl_dir, old_id_to_new_id)
    remap_label_class_ids(val_lbl_dir, old_id_to_new_id)

    assert (train_img_dir / "train.jpg").exists()
    assert (val_img_dir / "val.jpg").exists()
    assert (train_lbl_dir / "train.txt").exists()
    assert (val_lbl_dir / "val.txt").exists()
    assert class_mapping["bei_jing"] in {
        int(line.split()[0]) for line in (train_lbl_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    }
    assert class_mapping["bei_jing"] in {
        int(line.split()[0]) for line in (val_lbl_dir / "val.txt").read_text(encoding="utf-8").splitlines()
    }
