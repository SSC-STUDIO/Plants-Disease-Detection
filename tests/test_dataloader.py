"""Tests for dataset file discovery."""

from pathlib import Path

from config import DefaultConfigs
from dataset.dataloader import get_files


def test_get_files_deduplicates_case_insensitive_glob_matches(temp_dir):
    class_dir = temp_dir / "train" / "0"
    class_dir.mkdir(parents=True)
    image_path = class_dir / "sample.JPG"
    image_path.write_bytes(b"not-a-real-image")

    cfg = DefaultConfigs()
    cfg.paths.data_dir = str(temp_dir).replace("\\", "/")
    cfg.image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files = get_files(str(temp_dir / "train"), mode="train", cfg=cfg)

    assert len(files) == 1
    assert Path(files.iloc[0]["filename"]).name.lower() == "sample.jpg"
    assert files.iloc[0]["label"] == 0
