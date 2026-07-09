"""Tests for libs.image_security — format validation and secure loading."""
import io
import pytest
from PIL import Image

from libs.image_security import (
    ImageFormatError,
    ImageSecurityError,
    ImageTooLargeError,
    ImageValidationError,
    SecureImageLoader,
    get_secure_loader,
    validate_image_safe,
)


@pytest.fixture
def loader():
    """Create a SecureImageLoader with relaxed limits for testing."""
    return SecureImageLoader(
        max_pixels=10_000_000,
        max_dimension=5_000,
        max_file_size=10 * 1024 * 1024,
        min_file_size=1,
    )


@pytest.fixture
def tiny_png(tmp_path):
    """Save a 50×50 PNG to a temp file and return its path."""
    path = tmp_path / "test.png"
    Image.new("RGB", (50, 50), color="red").save(path, format="PNG")
    return str(path)


@pytest.fixture
def tiny_jpg(tmp_path):
    """Save a 50×50 JPEG to a temp file and return its path."""
    path = tmp_path / "test.jpg"
    Image.new("RGB", (50, 50), color="blue").save(path, format="JPEG")
    return str(path)


@pytest.fixture
def unsupported_file(tmp_path):
    """Create a fake TIFF file (unsupported format) and return its path."""
    path = tmp_path / "test.tiff"
    Image.new("RGB", (50, 50), color="green").save(path, format="TIFF")
    return str(path)


# --- validate_image_format ---

class TestValidateImageFormat:
    def test_accepts_png(self, loader, tiny_png):
        result = loader.validate_image_format(tiny_png)
        assert result == "PNG"

    def test_accepts_jpg(self, loader, tiny_jpg):
        result = loader.validate_image_format(tiny_jpg)
        assert result == "JPG"

    def test_rejects_unsupported(self, loader, unsupported_file):
        with pytest.raises(ImageFormatError, match="Unsupported image format"):
            loader.validate_image_format(unsupported_file)

    def test_rejects_unknown_extension(self, loader, tmp_path):
        path = tmp_path / "script.exe"
        path.write_bytes(b"MZ\x90\x00")
        with pytest.raises(ImageFormatError, match="Unsupported image format"):
            loader.validate_image_format(str(path))


# --- load_image format enforcement ---

class TestLoadImageFormatEnforcement:
    def test_load_image_rejects_unsupported_format(self, loader, unsupported_file):
        with pytest.raises(ImageFormatError, match="Unsupported image format"):
            loader.load_image(unsupported_file)

    def test_load_image_accepts_png(self, loader, tiny_png):
        img = loader.load_image(tiny_png)
        assert img.mode == "RGB"
        assert img.size == (50, 50)

    def test_load_image_accepts_jpg(self, loader, tiny_jpg):
        img = loader.load_image(tiny_jpg)
        assert img.mode == "RGB"
        assert img.size == (50, 50)

    def test_security_error_preserved_not_wrapped(self, loader, unsupported_file):
        """ImageFormatError should propagate as-is, not be wrapped in ImageValidationError."""
        with pytest.raises(ImageFormatError):
            loader.load_image(unsupported_file)


# --- load_image_from_bytes ---

class TestLoadImageFromBytes:
    def test_rejects_unsupported_bytes(self, loader):
        buf = io.BytesIO()
        Image.new("RGB", (50, 50)).save(buf, format="TIFF")
        with pytest.raises(ImageFormatError):
            loader.load_image_from_bytes(buf.getvalue())

    def test_accepts_png_bytes(self, loader):
        buf = io.BytesIO()
        Image.new("RGB", (50, 50)).save(buf, format="PNG")
        img = loader.load_image_from_bytes(buf.getvalue())
        assert img.size == (50, 50)


# --- validate_file_size ---

class TestValidateFileSize:
    def test_rejects_zero_byte_file(self, loader, tmp_path):
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        with pytest.raises(ImageValidationError, match="too small"):
            loader.validate_file_size(str(path))

    def test_rejects_oversized_file(self, loader, tmp_path):
        path = tmp_path / "huge.png"
        path.write_bytes(b"\x00" * (20 * 1024 * 1024))  # 20MB > 10MB limit
        with pytest.raises(ImageTooLargeError, match="too large"):
            loader.validate_file_size(str(path))


# --- validate_image_safe standalone function ---

class TestValidateImageSafe:
    def test_returns_true_for_safe_image(self, tiny_png):
        assert validate_image_safe(tiny_png) is True

    def test_returns_false_for_nonexistent(self):
        assert validate_image_safe("/nonexistent/path.png") is False

    def test_returns_false_for_unsupported_extension(self, unsupported_file):
        assert validate_image_safe(unsupported_file) is False


# --- get_secure_loader singleton ---

class TestGetSecureLoader:
    def test_returns_same_instance(self):
        a = get_secure_loader()
        b = get_secure_loader()
        assert a is b

    def test_returns_secure_image_loader(self):
        assert isinstance(get_secure_loader(), SecureImageLoader)
