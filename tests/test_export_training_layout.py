import csv
from argparse import Namespace

from PIL import Image

from tools.dataset_collector.export_training_layout import export_training_layout


def write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (80, 140, 60)).save(path)


def write_metadata(path):
    rows = [
        {
            "file_name": "data/train/0000_leaf/train.png",
            "split": "train",
            "label": "0",
            "label_name": "leaf",
            "source": "test",
            "source_url": "",
            "license": "MIT",
            "original_path": "train.png",
        },
        {
            "file_name": "data/test/0000_leaf/test.png",
            "split": "test",
            "label": "0",
            "label_name": "leaf",
            "source": "test",
            "source_url": "",
            "license": "MIT",
            "original_path": "test.png",
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_export_training_layout_maps_hf_splits_to_numeric_dirs(temp_dir):
    input_dir = temp_dir / "hf"
    output_dir = temp_dir / "numeric"
    write_image(input_dir / "data" / "train" / "0000_leaf" / "train.png")
    write_image(input_dir / "data" / "test" / "0000_leaf" / "test.png")
    write_metadata(input_dir / "metadata.csv")

    manifest = export_training_layout(
        Namespace(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_csv=input_dir / "metadata.csv",
            split_map=[],
            copy_mode="copy",
            overwrite=True,
            progress_interval=0,
        ),
    )

    assert manifest["split_counts"] == {"train": 1, "val": 1}
    assert (output_dir / "train" / "0" / "train.png").exists()
    assert (output_dir / "val" / "0" / "test.png").exists()


def test_export_training_layout_can_make_stratified_split(temp_dir):
    input_dir = temp_dir / "hf"
    output_dir = temp_dir / "numeric"
    rows = []
    for label in range(2):
        for index in range(5):
            file_name = f"data/train/{label:04d}_leaf/{label}_{index}.png"
            write_image(input_dir / file_name)
            rows.append(
                {
                    "file_name": file_name,
                    "split": "train",
                    "label": str(label),
                    "label_name": f"leaf-{label}",
                    "source": "test",
                    "source_url": "",
                    "license": "MIT",
                    "original_path": file_name,
                }
            )

    with open(input_dir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = export_training_layout(
        Namespace(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_csv=input_dir / "metadata.csv",
            split_map=[],
            stratified_val_ratio=0.4,
            stratified_seed=888,
            stratified_min_val_per_class=1,
            copy_mode="copy",
            overwrite=True,
            progress_interval=0,
        ),
    )

    assert manifest["split_counts"] == {"train": 6, "val": 4}
    assert len(list((output_dir / "val" / "0").glob("*.png"))) == 2
    assert len(list((output_dir / "val" / "1").glob("*.png"))) == 2
