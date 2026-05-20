import csv
import shutil

from PIL import Image

from tools.dataset_collector.filter_dataset_quality import FilterOptions, run_filter


def write_gradient_image(path, size=(128, 128), tweak_pixel=False):
    image = Image.new("RGB", size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, ((x + y) * 2) % 256)
    if tweak_pixel:
        pixels[0, 0] = (255, 0, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_metadata(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "split", "label", "label_name", "source", "source_url", "license", "original_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_filter_removes_low_quality_and_exact_duplicates(temp_dir):
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "filtered"
    good = input_dir / "data" / "train" / "0000_leaf" / "good.png"
    duplicate = input_dir / "data" / "train" / "0000_leaf" / "duplicate.png"
    low_quality = input_dir / "data" / "train" / "0000_leaf" / "flat.png"

    write_gradient_image(good)
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(good, duplicate)
    Image.new("RGB", (128, 128), (20, 20, 20)).save(low_quality)

    rows = [
        {"file_name": "data/train/0000_leaf/good.png", "split": "train", "label": "0", "label_name": "leaf", "source": "test", "source_url": "", "license": "MIT", "original_path": "good.png"},
        {"file_name": "data/train/0000_leaf/duplicate.png", "split": "train", "label": "0", "label_name": "leaf", "source": "test", "source_url": "", "license": "MIT", "original_path": "duplicate.png"},
        {"file_name": "data/train/0000_leaf/flat.png", "split": "train", "label": "0", "label_name": "leaf", "source": "test", "source_url": "", "license": "MIT", "original_path": "flat.png"},
    ]
    write_metadata(input_dir / "metadata.csv", rows)

    report = run_filter(
        FilterOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_csv=input_dir / "metadata.csv",
            copy_mode="copy",
            overwrite=True,
            min_file_size=10,
            min_dimension=64,
            min_stddev=2.0,
            min_entropy=1.0,
        ),
    )

    assert report["kept_images"] == 1
    assert report["rejection_counts"]["exact_duplicate"] == 1
    assert report["rejection_counts"]["low_stddev"] == 1
    assert (output_dir / "data" / "train" / "0000_leaf" / "good.png").exists()
    assert not (output_dir / "data" / "train" / "0000_leaf" / "duplicate.png").exists()


def test_filter_removes_near_duplicates_by_label(temp_dir):
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "filtered"
    original = input_dir / "data" / "train" / "0000_leaf" / "original.png"
    near = input_dir / "data" / "train" / "0000_leaf" / "near.png"

    write_gradient_image(original)
    write_gradient_image(near, tweak_pixel=True)
    write_metadata(
        input_dir / "metadata.csv",
        [
            {"file_name": "data/train/0000_leaf/original.png", "split": "train", "label": "0", "label_name": "leaf", "source": "test", "source_url": "", "license": "MIT", "original_path": "original.png"},
            {"file_name": "data/train/0000_leaf/near.png", "split": "train", "label": "0", "label_name": "leaf", "source": "test", "source_url": "", "license": "MIT", "original_path": "near.png"},
        ],
    )

    report = run_filter(
        FilterOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_csv=input_dir / "metadata.csv",
            copy_mode="copy",
            overwrite=True,
            min_file_size=10,
            min_dimension=64,
            min_stddev=2.0,
            min_entropy=1.0,
            near_duplicate_hamming=0,
        ),
    )

    assert report["kept_images"] == 2

    output_dir = temp_dir / "filtered_near"
    report = run_filter(
        FilterOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            metadata_csv=input_dir / "metadata.csv",
            copy_mode="copy",
            overwrite=True,
            min_file_size=10,
            min_dimension=64,
            min_stddev=2.0,
            min_entropy=1.0,
            near_duplicate_hamming=1,
        ),
    )

    assert report["kept_images"] == 1
    assert report["rejection_counts"]["near_duplicate"] == 1
