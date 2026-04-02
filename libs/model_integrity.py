"""
模型文件完整性验证模块
提供模型文件的哈希计算和验证功能，确保模型未被篡改
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger('ModelIntegrity')


@dataclass
class ModelHashRecord:
    """模型哈希记录"""
    file_path: str
    file_name: str
    file_size: int
    created_at: str
    
    # 多种哈希算法
    md5: str
    sha256: str
    sha512: str
    blake2b: str
    
    # 额外元数据
    model_architecture: Optional[str] = None
    training_epoch: Optional[int] = None
    accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelHashRecord':
        """从字典创建实例"""
        # 过滤掉不存在的字段
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


class ModelIntegrityVerifier:
    """模型完整性验证器
    
    提供模型文件的哈希计算和验证功能，防止:
    - 模型文件被恶意篡改
    - 模型文件传输过程中的损坏
    - 加载错误的模型版本
    """
    
    # 支持的哈希算法
    HASH_ALGORITHMS = {
        'md5': hashlib.md5,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
        'blake2b': lambda: hashlib.blake2b(digest_size=64),
    }
    
    def __init__(self, hash_store_path: Optional[str] = None, chunk_size: int = 8192):
        """初始化验证器
        
        Args:
            hash_store_path: 哈希记录存储路径
            chunk_size: 读取文件的块大小
        """
        self.chunk_size = chunk_size
        self.hash_store_path = hash_store_path or "model_hashes.json"
        self._hash_cache: Dict[str, ModelHashRecord] = {}
        self._load_hash_cache()
    
    def _load_hash_cache(self):
        """从文件加载哈希缓存"""
        if os.path.exists(self.hash_store_path):
            try:
                with open(self.hash_store_path, 'r') as f:
                    data = json.load(f)
                    for path, record_data in data.items():
                        self._hash_cache[path] = ModelHashRecord.from_dict(record_data)
                logger.info(f"Loaded {len(self._hash_cache)} hash records from {self.hash_store_path}")
            except Exception as e:
                logger.warning(f"Failed to load hash cache: {e}")
                self._hash_cache = {}
    
    def _save_hash_cache(self):
        """保存哈希缓存到文件"""
        try:
            # 转换为可序列化的字典
            data = {
                path: record.to_dict() 
                for path, record in self._hash_cache.items()
            }
            with open(self.hash_store_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save hash cache: {e}")
    
    def compute_file_hash(
        self,
        file_path: str,
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """计算文件的多种哈希值
        
        Args:
            file_path: 文件路径
            algorithms: 要计算的哈希算法列表，默认全部
            
        Returns:
            哈希算法名到哈希值的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        algorithms = algorithms or list(self.HASH_ALGORITHMS.keys())
        
        # 初始化哈希对象
        hash_objects = {
            alg: self.HASH_ALGORITHMS[alg]()
            for alg in algorithms
        }
        
        # 分块读取文件并计算哈希
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                for hash_obj in hash_objects.values():
                    hash_obj.update(chunk)
        
        # 返回哈希值
        return {
            alg: hash_obj.hexdigest()
            for alg, hash_obj in hash_objects.items()
        }
    
    def compute_model_hash(
        self,
        model_path: str,
        model_architecture: Optional[str] = None,
        training_epoch: Optional[int] = None,
        accuracy: Optional[float] = None
    ) -> ModelHashRecord:
        """计算模型的完整哈希记录
        
        Args:
            model_path: 模型文件路径
            model_architecture: 模型架构名称
            training_epoch: 训练轮次
            accuracy: 模型准确率
            
        Returns:
            模型哈希记录
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 计算文件信息
        stat = path.stat()
        file_size = stat.st_size
        created_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # 计算哈希
        hashes = self.compute_file_hash(model_path)
        
        record = ModelHashRecord(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_size=file_size,
            created_at=created_at,
            md5=hashes['md5'],
            sha256=hashes['sha256'],
            sha512=hashes['sha512'],
            blake2b=hashes['blake2b'],
            model_architecture=model_architecture,
            training_epoch=training_epoch,
            accuracy=accuracy
        )
        
        # 缓存记录
        self._hash_cache[str(path.absolute())] = record
        self._save_hash_cache()
        
        return record
    
    def verify_model_integrity(
        self,
        model_path: str,
        expected_hash: Optional[str] = None,
        hash_algorithm: str = 'sha256'
    ) -> Tuple[bool, Optional[str]]:
        """验证模型文件完整性
        
        Args:
            model_path: 模型文件路径
            expected_hash: 预期的哈希值（如果提供）
            hash_algorithm: 使用的哈希算法
            
        Returns:
            (是否通过验证, 错误信息)
        """
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
        
        # 如果提供了预期哈希值，直接比较
        if expected_hash:
            if hash_algorithm not in self.HASH_ALGORITHMS:
                return False, f"Unsupported hash algorithm: {hash_algorithm}"
            
            current_hash = self.compute_file_hash(model_path, [hash_algorithm])[hash_algorithm]
            
            if current_hash != expected_hash:
                return False, (
                    f"Hash mismatch! Expected {hash_algorithm}: {expected_hash}, "
                    f"but got: {current_hash}"
                )
            
            return True, None
        
        # 否则检查缓存的记录
        abs_path = str(Path(model_path).absolute())
        if abs_path in self._hash_cache:
            cached_record = self._hash_cache[abs_path]
            
            # 首先检查文件大小（快速检查）
            current_size = os.path.getsize(model_path)
            if current_size != cached_record.file_size:
                return False, (
                    f"File size mismatch! Expected: {cached_record.file_size}, "
                    f"but got: {current_size}"
                )
            
            # 然后检查SHA256
            current_sha256 = self.compute_file_hash(model_path, ['sha256'])['sha256']
            if current_sha256 != cached_record.sha256:
                return False, (
                    f"SHA256 mismatch! Model file may be corrupted or tampered. "
                    f"Expected: {cached_record.sha256}, but got: {current_sha256}"
                )
            
            return True, None
        
        # 没有缓存记录也没有预期哈希，计算并返回当前哈希
        current_hashes = self.compute_file_hash(model_path, ['sha256'])
        return True, f"No cached record found. Current SHA256: {current_hashes['sha256']}"
    
    def register_model(
        self,
        model_path: str,
        model_architecture: Optional[str] = None,
        training_epoch: Optional[int] = None,
        accuracy: Optional[float] = None,
        force: bool = False
    ) -> ModelHashRecord:
        """注册模型到完整性验证系统
        
        Args:
            model_path: 模型文件路径
            model_architecture: 模型架构名称
            training_epoch: 训练轮次
            accuracy: 模型准确率
            force: 是否强制重新计算（即使已有记录）
            
        Returns:
            模型哈希记录
        """
        abs_path = str(Path(model_path).absolute())
        
        if not force and abs_path in self._hash_cache:
            logger.info(f"Model already registered: {model_path}")
            return self._hash_cache[abs_path]
        
        logger.info(f"Registering model: {model_path}")
        record = self.compute_model_hash(
            model_path=model_path,
            model_architecture=model_architecture,
            training_epoch=training_epoch,
            accuracy=accuracy
        )
        
        logger.info(f"Model registered with SHA256: {record.sha256}")
        return record
    
    def get_model_hash(self, model_path: str) -> Optional[ModelHashRecord]:
        """获取模型的哈希记录
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            模型哈希记录，如果不存在则返回None
        """
        abs_path = str(Path(model_path).absolute())
        return self._hash_cache.get(abs_path)
    
    def list_registered_models(self) -> List[ModelHashRecord]:
        """列出所有已注册的模型
        
        Returns:
            模型哈希记录列表
        """
        return list(self._hash_cache.values())
    
    def remove_model_record(self, model_path: str) -> bool:
        """移除模型哈希记录
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否成功移除
        """
        abs_path = str(Path(model_path).absolute())
        if abs_path in self._hash_cache:
            del self._hash_cache[abs_path]
            self._save_hash_cache()
            return True
        return False
    
    def export_hash_report(self, output_path: str):
        """导出哈希报告到文件
        
        Args:
            output_path: 输出文件路径
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_models": len(self._hash_cache),
            "models": [
                {
                    "file_name": record.file_name,
                    "file_path": record.file_path,
                    "file_size": record.file_size,
                    "sha256": record.sha256,
                    "md5": record.md5,
                    "created_at": record.created_at,
                    "model_architecture": record.model_architecture,
                    "training_epoch": record.training_epoch,
                    "accuracy": record.accuracy
                }
                for record in self._hash_cache.values()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Hash report exported to: {output_path}")


# 全局验证器实例
_global_verifier: Optional[ModelIntegrityVerifier] = None

def get_integrity_verifier(hash_store_path: Optional[str] = None) -> ModelIntegrityVerifier:
    """获取全局完整性验证器实例"""
    global _global_verifier
    if _global_verifier is None:
        _global_verifier = ModelIntegrityVerifier(hash_store_path=hash_store_path)
    return _global_verifier


def verify_model_before_loading(
    model_path: str,
    hash_store_path: Optional[str] = None
) -> bool:
    """加载前验证模型完整性
    
    便捷函数，用于在加载模型前快速验证
    
    Args:
        model_path: 模型文件路径
        hash_store_path: 哈希存储路径
        
    Returns:
        是否通过验证
    """
    verifier = get_integrity_verifier(hash_store_path)
    is_valid, error_msg = verifier.verify_model_integrity(model_path)
    
    if not is_valid:
        logger.error(f"Model integrity check failed: {error_msg}")
        return False
    
    if error_msg:
        logger.warning(error_msg)
    else:
        logger.info(f"Model integrity verified: {model_path}")
    
    return True


def register_trained_model(
    model_path: str,
    model_architecture: str,
    training_epoch: int,
    accuracy: float,
    hash_store_path: Optional[str] = None
) -> ModelHashRecord:
    """注册训练完成的模型
    
    便捷函数，用于训练完成后注册模型哈希
    
    Args:
        model_path: 模型文件路径
        model_architecture: 模型架构名称
        training_epoch: 训练轮次
        accuracy: 模型准确率
        hash_store_path: 哈希存储路径
        
    Returns:
        模型哈希记录
    """
    verifier = get_integrity_verifier(hash_store_path)
    return verifier.register_model(
        model_path=model_path,
        model_architecture=model_architecture,
        training_epoch=training_epoch,
        accuracy=accuracy
    )