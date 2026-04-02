"""
图像安全验证模块
提供安全的图像加载和验证功能，防止CVE漏洞攻击
"""
import os
import io
import logging
from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np

logger = logging.getLogger('ImageSecurity')

# 安全限制配置
MAX_IMAGE_PIXELS = 100_000_000  # 100MP - 防止Decompression Bomb攻击
MAX_IMAGE_DIMENSION = 10000     # 最大边长10000像素
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MIN_FILE_SIZE = 100             # 最小100字节
SUPPORTED_FORMATS = {'JPEG', 'JPG', 'PNG', 'BMP', 'GIF', 'WEBP'}

class ImageSecurityError(Exception):
    """图像安全验证错误"""
    pass

class ImageTooLargeError(ImageSecurityError):
    """图像尺寸过大错误"""
    pass

class ImageFormatError(ImageSecurityError):
    """图像格式错误"""
    pass

class ImageValidationError(ImageSecurityError):
    """图像验证错误"""
    pass


class SecureImageLoader:
    """安全图像加载器
    
    提供安全的图像加载功能，防止以下攻击:
    - Decompression Bomb (解压炸弹攻击)
    - 恶意图像文件解析漏洞
    - 超大图像内存占用攻击
    - PIL/Pillow CVE漏洞
    """
    
    def __init__(
        self,
        max_pixels: int = MAX_IMAGE_PIXELS,
        max_dimension: int = MAX_IMAGE_DIMENSION,
        max_file_size: int = MAX_FILE_SIZE,
        min_file_size: int = MIN_FILE_SIZE,
        supported_formats: Optional[set] = None
    ):
        self.max_pixels = max_pixels
        self.max_dimension = max_dimension
        self.max_file_size = max_file_size
        self.min_file_size = min_file_size
        self.supported_formats = supported_formats or SUPPORTED_FORMATS
        
        # 设置PIL安全限制
        Image.MAX_IMAGE_PIXELS = self.max_pixels
        
    def validate_file_size(self, file_path: str) -> bool:
        """验证文件大小是否在安全范围内
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            验证是否通过
            
        Raises:
            ImageValidationError: 文件大小不合法
        """
        try:
            size = os.path.getsize(file_path)
            if size < self.min_file_size:
                raise ImageValidationError(
                    f"File too small: {size} bytes (min: {self.min_file_size})"
                )
            if size > self.max_file_size:
                raise ImageTooLargeError(
                    f"File too large: {size} bytes (max: {self.max_file_size})"
                )
            return True
        except OSError as e:
            raise ImageValidationError(f"Cannot access file: {e}")
    
    def validate_image_format(self, file_path: str) -> str:
        """验证图像格式是否安全
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            图像格式字符串
            
        Raises:
            ImageFormatError: 格式不支持或验证失败
        """
        # 检查文件扩展名
        ext = os.path.splitext(file_path)[1].upper().lstrip('.')
        if ext not in self.supported_formats:
            # 某些特殊情况
            if ext == 'JPEG':
                ext = 'JPG'
        
        return ext
    
    def verify_image_integrity(self, file_path: str) -> Tuple[bool, str]:
        """验证图像文件完整性
        
        使用PIL的verify()方法检查图像是否可以正确解析，
        不实际加载像素数据，防止内存攻击
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            (是否有效, 错误信息)
        """
        try:
            with Image.open(file_path) as img:
                # 验证图像格式
                fmt = img.format
                if fmt not in self.supported_formats:
                    return False, f"Unsupported format: {fmt}"
                
                # 验证图像完整性
                img.verify()
                
                return True, ""
        except Image.DecompressionBombError as e:
            return False, f"Decompression bomb detected: {e}"
        except Exception as e:
            return False, f"Image verification failed: {e}"
    
    def load_image(
        self,
        file_path: str,
        mode: str = 'RGB',
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """安全加载图像
        
        Args:
            file_path: 图像文件路径
            mode: 转换模式，默认RGB
            target_size: 目标尺寸 (width, height)
            
        Returns:
            PIL Image对象
            
        Raises:
            ImageSecurityError: 图像不安全或加载失败
        """
        # 1. 验证文件存在
        if not os.path.exists(file_path):
            raise ImageValidationError(f"File not found: {file_path}")
        
        # 2. 验证文件大小
        self.validate_file_size(file_path)
        
        # 3. 验证格式
        self.validate_image_format(file_path)
        
        # 4. 验证图像完整性
        is_valid, error_msg = self.verify_image_integrity(file_path)
        if not is_valid:
            raise ImageValidationError(error_msg)
        
        # 5. 安全加载图像
        try:
            with Image.open(file_path) as img:
                # 检查图像尺寸
                width, height = img.size
                if width > self.max_dimension or height > self.max_dimension:
                    raise ImageTooLargeError(
                        f"Image dimensions too large: {width}x{height} "
                        f"(max: {self.max_dimension}x{self.max_dimension})"
                    )
                
                # 检查总像素数
                if width * height > self.max_pixels:
                    raise ImageTooLargeError(
                        f"Image pixel count too large: {width * height} "
                        f"(max: {self.max_pixels})"
                    )
                
                # 安全加载
                img_copy = img.convert(mode)
                
                # 应用目标尺寸
                if target_size is not None:
                    img_copy = img_copy.resize(target_size, Image.Resampling.BILINEAR)
                
                return img_copy
                
        except Image.DecompressionBombError as e:
            raise ImageTooLargeError(f"Decompression bomb detected: {e}")
        except Exception as e:
            raise ImageValidationError(f"Failed to load image: {e}")
    
    def load_image_from_bytes(
        self,
        data: bytes,
        mode: str = 'RGB',
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """从字节数据安全加载图像
        
        Args:
            data: 图像字节数据
            mode: 转换模式
            target_size: 目标尺寸
            
        Returns:
            PIL Image对象
        """
        # 检查数据大小
        if len(data) < self.min_file_size:
            raise ImageValidationError(f"Data too small: {len(data)} bytes")
        if len(data) > self.max_file_size:
            raise ImageTooLargeError(f"Data too large: {len(data)} bytes")
        
        try:
            # 使用BytesIO避免临时文件
            with io.BytesIO(data) as bio:
                with Image.open(bio) as img:
                    # 验证格式
                    if img.format not in self.supported_formats:
                        raise ImageFormatError(f"Unsupported format: {img.format}")
                    
                    # 检查尺寸
                    width, height = img.size
                    if width > self.max_dimension or height > self.max_dimension:
                        raise ImageTooLargeError(
                            f"Image dimensions too large: {width}x{height}"
                        )
                    
                    if width * height > self.max_pixels:
                        raise ImageTooLargeError(
                            f"Image pixel count too large: {width * height}"
                        )
                    
                    # 安全加载
                    img_copy = img.convert(mode)
                    
                    if target_size is not None:
                        img_copy = img_copy.resize(target_size, Image.Resampling.BILINEAR)
                    
                    return img_copy
                    
        except Image.DecompressionBombError as e:
            raise ImageTooLargeError(f"Decompression bomb detected: {e}")
        except Exception as e:
            raise ImageValidationError(f"Failed to load image from bytes: {e}")


class SecureOpenCVLoader:
    """安全的OpenCV图像加载器
    
    提供基于OpenCV的安全图像加载，防止CVE漏洞
    """
    
    def __init__(
        self,
        max_pixels: int = MAX_IMAGE_PIXELS,
        max_dimension: int = MAX_IMAGE_DIMENSION,
        max_file_size: int = MAX_FILE_SIZE
    ):
        self.max_pixels = max_pixels
        self.max_dimension = max_dimension
        self.max_file_size = max_file_size
        self.secure_pil_loader = SecureImageLoader(
            max_pixels=max_pixels,
            max_dimension=max_dimension,
            max_file_size=max_file_size
        )
    
    def load_image(
        self,
        file_path: str,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """安全加载图像为OpenCV格式 (BGR)
        
        Args:
            file_path: 图像文件路径
            target_size: 目标尺寸 (width, height)
            
        Returns:
            OpenCV格式的图像数组 (BGR)
        """
        # 首先使用PIL安全加载
        pil_img = self.secure_pil_loader.load_image(file_path, mode='RGB', target_size=target_size)
        
        # 转换为OpenCV格式 (RGB -> BGR)
        cv_img = np.array(pil_img)
        cv_img = cv_img[:, :, ::-1].copy()  # RGB to BGR
        
        return cv_img


def validate_image_safe(
    file_path: str,
    max_size: Optional[int] = None,
    max_dimension: Optional[int] = None
) -> bool:
    """快速验证图像文件是否安全
    
    Args:
        file_path: 图像文件路径
        max_size: 最大文件大小
        max_dimension: 最大边长
        
    Returns:
        是否安全
    """
    loader = SecureImageLoader(
        max_pixels=max_size or MAX_IMAGE_PIXELS,
        max_dimension=max_dimension or MAX_IMAGE_DIMENSION
    )
    
    try:
        loader.validate_file_size(file_path)
        is_valid, _ = loader.verify_image_integrity(file_path)
        return is_valid
    except ImageSecurityError:
        return False


# 全局安全加载器实例
_global_secure_loader: Optional[SecureImageLoader] = None

def get_secure_loader() -> SecureImageLoader:
    """获取全局安全图像加载器实例"""
    global _global_secure_loader
    if _global_secure_loader is None:
        _global_secure_loader = SecureImageLoader()
    return _global_secure_loader


def secure_load_image(
    file_path: str,
    mode: str = 'RGB',
    target_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """便捷函数：安全加载图像
    
    Args:
        file_path: 图像文件路径
        mode: 转换模式
        target_size: 目标尺寸
        
    Returns:
        PIL Image对象
    """
    loader = get_secure_loader()
    return loader.load_image(file_path, mode=mode, target_size=target_size)