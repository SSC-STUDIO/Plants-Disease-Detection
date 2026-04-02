"""
训练数据安全验证模块
提供训练数据的安全验证和清洗功能，防止数据投毒和对抗样本攻击
"""
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger('DataValidation')


@dataclass
class DataValidationResult:
    """数据验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Optional[Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class DataSanitizer:
    """数据清洗器
    
    清洗和标准化训练数据，防止:
    - 路径遍历攻击
    - 恶意文件名
    - 异常数据值
    - 标签投毒
    """
    
    # 危险的文件名字符
    DANGEROUS_CHARS = set(['<', '>', ':', '"', '|', '?', '*', '\x00'])
    
    # 路径遍历模式
    PATH_TRAVERSAL_PATTERNS = [
        r'\.\./',           # ../
        r'\.\.\\',          # ..\
        r'%2e%2e/',         # URL编码的../
        r'%2e%2e%2f',       # URL编码的../
        r'\.\.\Z',         # ..在末尾
        r'^/',               # 绝对路径Unix
        r'^[a-zA-Z]:',       # 绝对路径Windows
    ]
    
    # 最大标签值（防止整数溢出）
    MAX_LABEL_VALUE = 999999
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path
        self.path_traversal_regex = re.compile(
            '|'.join(self.PATH_TRAVERSAL_PATTERNS),
            re.IGNORECASE
        )
    
    def sanitize_filename(self, filename: str) -> Tuple[str, Optional[str]]:
        """清洗文件名
        
        Args:
            filename: 原始文件名
            
        Returns:
            (清洗后的文件名, 错误信息)
        """
        if not filename:
            return "", "Empty filename"
        
        # 检查路径遍历
        if self.path_traversal_regex.search(filename):
            return "", f"Path traversal detected in filename: {filename}"
        
        # 检查危险字符
        for char in filename:
            if char in self.DANGEROUS_CHARS:
                return "", f"Dangerous character '{char}' in filename: {filename}"
        
        # 规范化路径
        filename = os.path.normpath(filename)
        
        # 移除前导的./和../
        while filename.startswith('../') or filename.startswith('..\\'):
            filename = filename[3:]
        while filename.startswith('./') or filename.startswith('.\\'):
            filename = filename[2:]
        
        # 确保文件名不为空
        if not filename or filename == '.' or filename == '..':
            return "", f"Invalid filename after sanitization: {filename}"
        
        return filename, None
    
    def validate_file_path(
        self,
        file_path: str,
        must_exist: bool = True,
        allowed_extensions: Optional[Set[str]] = None
    ) -> DataValidationResult:
        """验证文件路径
        
        Args:
            file_path: 文件路径
            must_exist: 是否要求文件必须存在
            allowed_extensions: 允许的扩展名集合
            
        Returns:
            验证结果
        """
        errors = []
        warnings = []
        
        if not file_path:
            return DataValidationResult(False, ["Empty file path"], [])
        
        # 清洗文件名
        sanitized_path, error = self.sanitize_filename(file_path)
        if error:
            return DataValidationResult(False, [error], [])
        
        # 验证扩展名
        if allowed_extensions:
            ext = os.path.splitext(sanitized_path)[1].lower()
            if ext not in allowed_extensions:
                return DataValidationResult(
                    False,
                    [f"File extension '{ext}' not allowed. Allowed: {allowed_extensions}"],
                    []
                )
        
        # 检查文件是否在基础目录内（如果指定了）
        if self.base_path:
            try:
                abs_base = os.path.abspath(self.base_path)
                abs_path = os.path.abspath(sanitized_path)
                
                # 检查路径是否在基础目录内
                if not abs_path.startswith(abs_base + os.sep) and abs_path != abs_base:
                    return DataValidationResult(
                        False,
                        [f"File path '{file_path}' is outside the allowed directory"],
                        []
                    )
            except Exception as e:
                return DataValidationResult(
                    False,
                    [f"Path validation error: {e}"],
                    []
                )
        
        # 检查文件是否存在
        if must_exist and not os.path.exists(sanitized_path):
            return DataValidationResult(
                False,
                [f"File does not exist: {sanitized_path}"],
                []
            )
        
        return DataValidationResult(True, [], [], sanitized_path)
    
    def validate_label(self, label: Union[int, str], num_classes: Optional[int] = None) -> DataValidationResult:
        """验证标签值
        
        Args:
            label: 标签值
            num_classes: 类别总数
            
        Returns:
            验证结果
        """
        errors = []
        
        # 转换为整数
        try:
            if isinstance(label, str):
                label = int(label.strip())
            elif isinstance(label, float):
                label = int(label)
            elif not isinstance(label, int):
                return DataValidationResult(
                    False,
                    [f"Invalid label type: {type(label)}"],
                    []
                )
        except (ValueError, TypeError) as e:
            return DataValidationResult(
                False,
                [f"Cannot convert label to integer: {label}, error: {e}"],
                []
            )
        
        # 检查范围
        if label < 0:
            return DataValidationResult(False, [f"Negative label value: {label}"], [])
        
        if label > self.MAX_LABEL_VALUE:
            return DataValidationResult(
                False,
                [f"Label value too large: {label} (max: {self.MAX_LABEL_VALUE})"],
                []
            )
        
        # 检查是否在有效类别范围内
        if num_classes is not None and label >= num_classes:
            return DataValidationResult(
                False,
                [f"Label {label} exceeds number of classes {num_classes}"],
                []
            )
        
        return DataValidationResult(True, [], [], label)
    
    def sanitize_dataset_record(
        self,
        record: Dict[str, Any],
        num_classes: Optional[int] = None,
        allowed_extensions: Optional[Set[str]] = None
    ) -> DataValidationResult:
        """清洗数据集记录
        
        Args:
            record: 数据集记录字典，包含 'filename' 和 'label'
            num_classes: 类别总数
            allowed_extensions: 允许的扩展名
            
        Returns:
            验证结果
        """
        errors = []
        warnings = []
        
        # 检查必需字段
        if 'filename' not in record:
            return DataValidationResult(False, ["Missing 'filename' field"], [])
        
        # 验证文件路径
        file_result = self.validate_file_path(
            record['filename'],
            allowed_extensions=allowed_extensions
        )
        if not file_result.is_valid:
            return DataValidationResult(False, file_result.errors, [])
        
        sanitized_record = {'filename': file_result.sanitized_data}
        
        # 验证标签
        if 'label' in record:
            label_result = self.validate_label(record['label'], num_classes)
            if not label_result.is_valid:
                return DataValidationResult(False, label_result.errors, [])
            sanitized_record['label'] = label_result.sanitized_data
        
        # 复制其他安全字段
        safe_fields = ['split', 'fold', 'metadata']
        for field in safe_fields:
            if field in record:
                sanitized_record[field] = record[field]
        
        return DataValidationResult(True, [], [], sanitized_record)


class DataQualityChecker:
    """数据质量检查器
    
    检查数据集的质量问题，包括:
    - 类别不平衡
    - 重复样本
    - 异常标签分布
    """
    
    def __init__(self, min_samples_per_class: int = 10, max_imbalance_ratio: float = 10.0):
        self.min_samples_per_class = min_samples_per_class
        self.max_imbalance_ratio = max_imbalance_ratio
    
    def check_class_balance(self, labels: List[int]) -> Tuple[bool, List[str], Dict[int, int]]:
        """检查类别平衡
        
        Args:
            labels: 标签列表
            
        Returns:
            (是否平衡, 警告信息, 类别统计)
        """
        if not labels:
            return False, ["Empty label list"], {}
        
        # 统计每个类别的样本数
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        warnings = []
        
        # 检查每个类别的最小样本数
        for label, count in class_counts.items():
            if count < self.min_samples_per_class:
                warnings.append(
                    f"Class {label} has only {count} samples "
                    f"(minimum recommended: {self.min_samples_per_class})"
                )
        
        # 检查类别不平衡
        if len(class_counts) > 1:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            
            if max_count / min_count > self.max_imbalance_ratio:
                warnings.append(
                    f"Severe class imbalance: max/min ratio = {max_count/min_count:.1f} "
                    f"(threshold: {self.max_imbalance_ratio})"
                )
        
        is_balanced = len(warnings) == 0
        return is_balanced, warnings, class_counts
    
    def check_duplicates(self, filenames: List[str]) -> Tuple[List[str], List[str]]:
        """检查重复样本
        
        Args:
            filenames: 文件名列表
            
        Returns:
            (重复文件名列表, 警告信息)
        """
        seen = set()
        duplicates = []
        
        for filename in filenames:
            abs_path = os.path.abspath(filename)
            if abs_path in seen:
                duplicates.append(filename)
            seen.add(abs_path)
        
        warnings = []
        if duplicates:
            warnings.append(f"Found {len(duplicates)} duplicate samples")
        
        return duplicates, warnings
    
    def check_for_adversarial_patterns(
        self,
        image_array: np.ndarray,
        max_pixel_value: float = 255.0
    ) -> DataValidationResult:
        """检查图像是否存在对抗样本特征
        
        检测一些常见的对抗样本模式:
        - 像素值异常（超出正常范围）
        - 过度锐化的边缘
        - 异常的高频噪声
        
        Args:
            image_array: 图像数组
            max_pixel_value: 最大像素值
            
        Returns:
            验证结果
        """
        warnings = []
        
        # 检查像素值范围
        if image_array.min() < 0 or image_array.max() > max_pixel_value:
            warnings.append(
                f"Pixel values out of range: [{image_array.min()}, {image_array.max()}]"
            )
        
        # 检查NaN和Inf
        if np.isnan(image_array).any():
            return DataValidationResult(
                False,
                ["Image contains NaN values"],
                [],
                None
            )
        
        if np.isinf(image_array).any():
            return DataValidationResult(
                False,
                ["Image contains infinite values"],
                [],
                None
            )
        
        # 检查异常的高频噪声（简单方差检查）
        pixel_variance = np.var(image_array)
        if pixel_variance > (max_pixel_value ** 2) * 0.5:
            warnings.append(f"Abnormally high pixel variance: {pixel_variance}")
        
        is_valid = len(warnings) == 0
        return DataValidationResult(is_valid, [], warnings, None)


class SecureDatasetLoader:
    """安全数据集加载器
    
    整合所有安全验证的数据集加载器
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        num_classes: Optional[int] = None,
        allowed_extensions: Optional[Set[str]] = None,
        min_samples_per_class: int = 10
    ):
        self.sanitizer = DataSanitizer(base_path=base_path)
        self.quality_checker = DataQualityChecker(min_samples_per_class=min_samples_per_class)
        self.num_classes = num_classes
        self.allowed_extensions = allowed_extensions or {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def validate_dataset_records(
        self,
        records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """验证并清洗数据集记录
        
        Args:
            records: 原始记录列表
            
        Returns:
            (清洗后的记录, 错误列表, 警告列表)
        """
        valid_records = []
        all_errors = []
        all_warnings = []
        
        for i, record in enumerate(records):
            result = self.sanitizer.sanitize_dataset_record(
                record,
                num_classes=self.num_classes,
                allowed_extensions=self.allowed_extensions
            )
            
            if result.is_valid:
                valid_records.append(result.sanitized_data)
                all_warnings.extend([f"Record {i}: {w}" for w in result.warnings])
            else:
                all_errors.extend([f"Record {i}: {e}" for e in result.errors])
        
        # 检查类别平衡
        if valid_records:
            labels = [r['label'] for r in valid_records if 'label' in r]
            if labels:
                _, warnings, _ = self.quality_checker.check_class_balance(labels)
                all_warnings.extend(warnings)
        
        # 检查重复
        filenames = [r['filename'] for r in valid_records]
        _, dup_warnings = self.quality_checker.check_duplicates(filenames)
        all_warnings.extend(dup_warnings)
        
        return valid_records, all_errors, all_warnings


# 便捷函数
def validate_data_path(file_path: str, base_path: Optional[str] = None) -> bool:
    """验证数据路径是否安全
    
    Args:
        file_path: 文件路径
        base_path: 基础路径（用于路径遍历检查）
        
    Returns:
        是否安全
    """
    sanitizer = DataSanitizer(base_path=base_path)
    result = sanitizer.validate_file_path(file_path)
    return result.is_valid


def sanitize_label(label: Union[int, str], num_classes: Optional[int] = None) -> Optional[int]:
    """清洗标签值
    
    Args:
        label: 原始标签
        num_classes: 类别总数
        
    Returns:
        清洗后的标签，无效则返回None
    """
    sanitizer = DataSanitizer()
    result = sanitizer.validate_label(label, num_classes)
    if result.is_valid:
        return result.sanitized_data
    return None