"""
路径安全验证工具模块
提供路径遍历防护、路径验证和安全文件操作功能
"""
import os
import re
from pathlib import Path
from typing import Optional, List, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class PathSecurityError(Exception):
    """路径安全错误异常"""
    pass


class PathValidator:
    """路径验证器类，提供各种路径安全验证功能"""
    
    # 危险路径模式 - 路径遍历攻击常用模式
    DANGEROUS_PATTERNS = [
        r'\.\.+',           # ../ 或 ..\\
        r'\.\.\\',         # ..\\ (Windows)
        r'%2e%2e',          # URL编码的 ..
        r'%252e%252e',      # 双重URL编码的 ..
        r'0x2e0x2e',        # 十六进制编码的 ..
        r'\.{2,}',          # 两个或多个点
        r'~',               # 波浪号展开
        r'\$[A-Z_]+',       # 环境变量
    ]
    
    # 敏感系统路径 - 禁止访问的路径
    SENSITIVE_PATHS = [
        '/etc/passwd',
        '/etc/shadow',
        '/etc/hosts',
        '/proc/',
        '/sys/',
        '/boot/',
        '/root/',
        '/var/log/',
        'C:\\Windows\\',
        'C:\\Program Files\\',
        'C:\\ProgramData\\',
    ]
    
    @classmethod
    def validate_path_traversal(cls, user_path: str) -> bool:
        """
        验证路径是否包含路径遍历攻击模式
        
        参数:
            user_path: 用户提供的文件路径
            
        返回:
            bool: True如果路径安全，False如果检测到攻击模式
        """
        if not user_path or not isinstance(user_path, str):
            return False
            
        # 检查危险模式
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, user_path, re.IGNORECASE):
                logger.warning(f"Path traversal pattern detected: {pattern} in {user_path}")
                return False
        
        # 解码并检查URL编码的路径遍历
        decoded_path = user_path.replace('%2f', '/').replace('%5c', '\\')
        if '..' in decoded_path:
            logger.warning(f"Decoded path traversal detected in: {user_path}")
            return False
            
        return True
    
    @classmethod
    def is_within_allowed_directory(cls, user_path: str, allowed_base: str) -> bool:
        """
        验证用户路径是否在允许的目录范围内
        
        参数:
            user_path: 用户提供的文件路径
            allowed_base: 允许的基础目录路径
            
        返回:
            bool: True如果在允许范围内，False否则
        """
        try:
            # 获取绝对路径
            base_path = Path(allowed_base).resolve()
            target_path = Path(user_path).resolve()
            
            # 检查目标路径是否在基础路径下
            try:
                target_path.relative_to(base_path)
                return True
            except ValueError:
                logger.warning(f"Path {target_path} is outside allowed directory {base_path}")
                return False
        except Exception as e:
            logger.error(f"Error validating path: {e}")
            return False
    
    @classmethod
    def sanitize_filename(cls, filename: str, max_length: int = 255) -> str:
        """
        清理文件名，移除危险字符
        
        参数:
            filename: 原始文件名
            max_length: 最大文件名长度
            
        返回:
            str: 清理后的文件名
        """
        if not filename:
            return "unnamed"
        
        # 移除路径分隔符和危险字符
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # 移除前导和尾随的空格和点
        sanitized = sanitized.strip(' .')
        
        # 限制长度
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:max_length - len(ext)] + ext
        
        # 确保不为空
        if not sanitized:
            sanitized = "unnamed"
            
        return sanitized
    
    @classmethod
    def validate_file_extension(cls, filepath: str, allowed_extensions: List[str]) -> bool:
        """
        验证文件扩展名是否在允许列表中
        
        参数:
            filepath: 文件路径
            allowed_extensions: 允许的扩展名列表 (如 ['.jpg', '.png'])
            
        返回:
            bool: True如果扩展名有效
        """
        ext = os.path.splitext(filepath)[1].lower()
        allowed = [e.lower() for e in allowed_extensions]
        return ext in allowed
    
    @classmethod
    def is_sensitive_path(cls, path: str) -> bool:
        """
        检查路径是否为敏感系统路径
        
        参数:
            path: 要检查的路径
            
        返回:
            bool: True如果是敏感路径
        """
        path_lower = path.lower()
        for sensitive in cls.SENSITIVE_PATHS:
            if path_lower.startswith(sensitive.lower()):
                return True
        return False
    
    @classmethod
    def safe_join(cls, base: str, *paths: str) -> str:
        """
        安全地拼接路径，防止路径遍历
        
        参数:
            base: 基础路径
            *paths: 要拼接的路径组件
            
        返回:
            str: 安全拼接后的路径
            
        抛出:
            PathSecurityError: 如果检测到路径遍历
        """
        # 验证每个路径组件
        for path in paths:
            if not cls.validate_path_traversal(path):
                raise PathSecurityError(f"Path traversal detected in component: {path}")
        
        # 安全拼接
        base_path = Path(base).resolve()
        result = base_path.joinpath(*paths).resolve()
        
        # 确保结果仍在基础路径下
        try:
            result.relative_to(base_path)
        except ValueError:
            raise PathSecurityError(f"Result path {result} escapes base directory {base_path}")
        
        return str(result)


def secure_open_file(
    filepath: str,
    mode: str = 'r',
    allowed_directories: Optional[List[str]] = None,
    allowed_extensions: Optional[List[str]] = None,
    encoding: Optional[str] = None
):
    """
    安全地打开文件，进行多层验证
    
    参数:
        filepath: 文件路径
        mode: 打开模式
        allowed_directories: 允许的目录列表
        allowed_extensions: 允许的文件扩展名列表
        encoding: 文件编码
        
    返回:
        file object: 打开的文件对象
        
    抛出:
        PathSecurityError: 如果安全验证失败
    """
    validator = PathValidator()
    
    # 验证路径遍历
    if not validator.validate_path_traversal(filepath):
        raise PathSecurityError(f"Path traversal detected: {filepath}")
    
    # 验证敏感路径
    if validator.is_sensitive_path(filepath):
        raise PathSecurityError(f"Access to sensitive path denied: {filepath}")
    
    # 验证扩展名
    if allowed_extensions:
        if not validator.validate_file_extension(filepath, allowed_extensions):
            raise PathSecurityError(
                f"File extension not allowed. Allowed: {allowed_extensions}"
            )
    
    # 验证目录范围
    if allowed_directories:
        in_allowed = any(
            validator.is_within_allowed_directory(filepath, allowed_dir)
            for allowed_dir in allowed_directories
        )
        if not in_allowed:
            raise PathSecurityError(
                f"Path {filepath} is outside allowed directories"
            )
    
    # 安全打开文件
    resolved_path = Path(filepath).resolve()
    
    if encoding:
        return open(resolved_path, mode, encoding=encoding)
    else:
        return open(resolved_path, mode)


def validate_model_path(model_path: str, allowed_extensions: List[str] = None) -> str:
    """
    验证模型文件路径的安全性
    
    参数:
        model_path: 模型文件路径
        allowed_extensions: 允许的扩展名列表 (默认: ['.pth', '.pt', '.pth.tar', '.onnx'])
        
    返回:
        str: 验证后的绝对路径
        
    抛出:
        PathSecurityError: 如果验证失败
    """
    if allowed_extensions is None:
        allowed_extensions = ['.pth', '.pt', '.pth.tar', '.onnx', '.ckpt', '.safetensors']
    
    validator = PathValidator()
    
    # 验证路径遍历
    if not validator.validate_path_traversal(model_path):
        raise PathSecurityError(f"Path traversal detected in model path: {model_path}")
    
    # 验证敏感路径
    if validator.is_sensitive_path(model_path):
        raise PathSecurityError(f"Access to sensitive path denied: {model_path}")
    
    # 验证扩展名
    if not validator.validate_file_extension(model_path, allowed_extensions):
        raise PathSecurityError(
            f"Invalid model file extension. Allowed: {allowed_extensions}"
        )
    
    # 检查文件存在性
    resolved_path = Path(model_path).resolve()
    if not resolved_path.exists():
        raise PathSecurityError(f"Model file not found: {model_path}")
    
    if not resolved_path.is_file():
        raise PathSecurityError(f"Model path is not a file: {model_path}")
    
    return str(resolved_path)


def validate_image_path(image_path: str, allowed_base_dir: Optional[str] = None) -> str:
    """
    验证图像文件路径的安全性
    
    参数:
        image_path: 图像文件路径
        allowed_base_dir: 允许的基础目录
        
    返回:
        str: 验证后的绝对路径
        
    抛出:
        PathSecurityError: 如果验证失败
    """
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif']
    
    validator = PathValidator()
    
    # 验证路径遍历
    if not validator.validate_path_traversal(image_path):
        raise PathSecurityError(f"Path traversal detected in image path: {image_path}")
    
    # 验证敏感路径
    if validator.is_sensitive_path(image_path):
        raise PathSecurityError(f"Access to sensitive path denied: {image_path}")
    
    # 验证扩展名
    if not validator.validate_file_extension(image_path, allowed_extensions):
        raise PathSecurityError(
            f"Invalid image file extension. Allowed: {allowed_extensions}"
        )
    
    # 验证在允许目录内
    if allowed_base_dir:
        if not validator.is_within_allowed_directory(image_path, allowed_base_dir):
            raise PathSecurityError(
                f"Image path {image_path} is outside allowed directory {allowed_base_dir}"
            )
    
    # 检查文件存在性
    resolved_path = Path(image_path).resolve()
    if not resolved_path.exists():
        raise PathSecurityError(f"Image file not found: {image_path}")
    
    return str(resolved_path)


def safe_makedirs(path: str, mode: int = 0o755, allowed_base: Optional[str] = None) -> None:
    """
    安全地创建目录
    
    参数:
        path: 目录路径
        mode: 目录权限
        allowed_base: 允许创建的基础目录
        
    抛出:
        PathSecurityError: 如果验证失败
    """
    validator = PathValidator()
    
    # 验证路径遍历
    if not validator.validate_path_traversal(path):
        raise PathSecurityError(f"Path traversal detected in directory path: {path}")
    
    # 验证敏感路径
    if validator.is_sensitive_path(path):
        raise PathSecurityError(f"Cannot create directory in sensitive path: {path}")
    
    # 验证在允许目录内
    if allowed_base:
        resolved_base = Path(allowed_base).resolve()
        resolved_path = Path(path).resolve()
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise PathSecurityError(
                f"Directory path {path} is outside allowed base {allowed_base}"
            )
    
    # 创建目录
    os.makedirs(path, mode=mode, exist_ok=True)


# 便捷函数
def is_safe_path(user_path: str, base_path: str) -> bool:
    """
    检查用户路径是否安全（在基础路径范围内且无遍历）
    
    参数:
        user_path: 用户提供的文件路径
        base_path: 允许的基础目录
        
    返回:
        bool: True如果路径安全
    """
    validator = PathValidator()
    if not validator.validate_path_traversal(user_path):
        return False
    return validator.is_within_allowed_directory(user_path, base_path)


def sanitize_path_component(component: str) -> str:
    """
    清理路径组件（目录名或文件名）
    
    参数:
        component: 路径组件
        
    返回:
        str: 清理后的组件
    """
    return PathValidator.sanitize_filename(component)
