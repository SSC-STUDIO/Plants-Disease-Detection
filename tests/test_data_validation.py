"""Tests for data validation path handling."""

from libs.data_validation import DataSanitizer


def test_validate_file_path_allows_absolute_path_inside_base(temp_dir):
    image_path = temp_dir / "train" / "0" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")

    sanitizer = DataSanitizer(base_path=str(temp_dir))
    result = sanitizer.validate_file_path(str(image_path), must_exist=True, allowed_extensions={".jpg"})

    assert result.is_valid
    assert result.sanitized_data


def test_validate_file_path_rejects_absolute_path_outside_base(temp_dir):
    base = temp_dir / "base"
    outside = temp_dir / "outside" / "sample.jpg"
    base.mkdir()
    outside.parent.mkdir()
    outside.write_bytes(b"image")

    sanitizer = DataSanitizer(base_path=str(base))
    result = sanitizer.validate_file_path(str(outside), must_exist=True, allowed_extensions={".jpg"})

    assert not result.is_valid
