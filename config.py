from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, field, fields
import os

@dataclass
class PathConfig:
    """路径配置类，包含所有文件路径和目录"""
    
    # 基础目录
    base_dir: str = "./"
    data_dir: str = "./data/"
    
    # 数据目录
    train_dir: str = "./data/train/"
    train_images_dir: str = "./data/train/images/"
    test_dir: str = "./data/test/"
    test_images_dir: str = "./data/test/images/"
    val_dir: str = "./data/val/"
    temp_dir: str = "./data/temp/"
    temp_images_dir: str = "./data/temp/images/"
    temp_labels_dir: str = "./data/temp/labels/"
    temp_dataset_dir: str = "./data/temp/dataset/"
    
    # 合并数据集目录
    merged_train_dir: str = "./data/merged_train/"
    merged_test_dir: str = "./data/merged_test/"
    merged_val_dir: str = "./data/merged_val/"
    
    # 数据增强目录
    aug_dir: str = "./data/aug/"
    aug_train_dir: str = "./data/aug/train/"
    augmented_images_dir: str = "./data/aug/images/"
    
    # 检查点目录
    weight_dir: str = "./checkpoints/"
    best_weight_dir: str = "./checkpoints/best/"
    
    # 输出目录
    submit_dir: str = "./submit/"
    log_dir: str = "./log/"
    report_dir: str = "./reports/"
    
    # 日志文件
    training_log: str = "./log/training.log"
    data_aug_log: str = "./log/data_augmentation.log"
    data_proc_log: str = "./log/data_processing.log"
    inference_log: str = "./log/inference.log"
    utils_log: str = "./log/utils.log"
    
    # 数据文件
    train_dataset_folder_name: str = "AgriculturalDisease_trainingset"
    val_dataset_folder_name: str = "AgriculturalDisease_validationset"
    train_annotation: str = "./data/temp/labels/AgriculturalDisease_train_annotations.json"
    val_annotation: str = "./data/temp/labels/AgriculturalDisease_validation_annotations.json"
    prediction_file: str = "./submit/prediction.json"
    
    def __post_init__(self):
        """确保所有路径都有一致的格式并创建必要的目录"""
        # 规范化路径格式
        for config_field in fields(self):
            attr_name = config_field.name
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str) and attr_name.endswith(('_dir', '_path')):
                # 确保目录路径以斜杠结尾
                if attr_name.endswith('_dir') and not attr_value.endswith('/'):
                    attr_value = attr_value + '/'
                
                # 确保所有路径使用正斜杠，避免Windows路径问题
                attr_value = attr_value.replace('\\', '/')
                
                setattr(self, attr_name, attr_value)
        
        # 创建关键目录
        essential_dirs = [
            self.data_dir,
            self.log_dir,
            self.report_dir,
            self.train_dir,
            self.test_dir,
            self.test_images_dir,
            self.val_dir,
            self.temp_dir,
            self.temp_images_dir, 
            self.temp_labels_dir,
            self.temp_dataset_dir,
            self.weight_dir,
            self.best_weight_dir,
            self.submit_dir,
            self.aug_dir,
            self.aug_train_dir,
            self.augmented_images_dir
        ]
        
        for directory in essential_dirs:
            try:
                # 确保路径格式一致后再创建目录
                dir_path = directory.replace('\\', '/')
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {str(e)}")

def get_path_config():
    """创建PathConfig实例的工厂函数"""
    return PathConfig()

@dataclass
class DefaultConfigs:
    """配置类，包含所有训练和数据处理的参数设置"""
    
    # 创建路径配置 - 使用default_factory避免可变默认值的问题
    paths: PathConfig = field(default_factory=get_path_config)
    
    # 以下字段需要在__post_init__中初始化，因为它们依赖于paths
    train_data: str = field(default="")  # 训练数据路径
    test_data: str = field(default="")  # 测试数据路径
    val_data: str = "none"  # 验证数据路径
    model_name: str = "convnextv2_base_384"  # 使用更现代的 ConvNeXt V2 Base 384 模型
    weights: str = field(default="")  # 权重保存路径
    best_weights: str = field(default="")  # 最佳模型保存路径
    submit: str = field(default="")  # 提交结果保存路径
    logs: str = field(default="")  # 日志保存路径
    
    # 数据集路径配置 Dataset Path Configuration
    dataset_path: Optional[str] = None  # 外部数据集路径，设置为None表示使用默认的data目录
    training_dataset_file: str = "ai_challenger_pdr2018_trainingset_20181023.zip"  # 训练集数据文件名
    validation_dataset_file: str = "ai_challenger_pdr2018_validationset_20181023.zip"  # 验证集数据文件名
    use_custom_dataset_path: bool = False  # 是否使用自定义数据集路径
    supported_dataset_formats: Tuple[str, ...] = ('.zip', '.rar', '.tar', '.gz', '.tgz')  # 支持的数据集格式
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')  # 允许的图像扩展名
    
    # 数据集合并配置 Dataset Merging Configuration
    merge_datasets: bool = False  # 是否合并多个数据集，当设置时会影响所有三个数据集的合并设置
    merge_train_datasets: bool = False  # 是否合并训练数据集
    merge_val_datasets: bool = False  # 是否合并验证数据集
    merge_test_datasets: bool = False  # 是否合并测试数据集，默认开启
    dataset_to_use: str = "auto"  # 不合并时选择使用哪个数据集: "auto"(最大的), "first", "last", "specific"
    specific_dataset: str = ""  # 指定使用的数据集名称，当dataset_to_use="specific"时有效
    duplicate_test_to_common: bool = False  # 是否将测试集复制到通用测试目录
    merge_force: bool = False  # 是否强制重新合并已存在的数据集
    merge_on_startup: bool = False  # 是否在程序启动时自动合并数据集
    force_data_processing: bool = False  # 是否强制重新处理数据（即使已存在处理后的数据）
    force_augmentation: bool = False  # 是否强制重新生成数据增强（即使已存在增强数据）
    force_merge: bool = False  # 是否强制重新合并数据集（即使合并的数据集已存在）

    # 测试集处理配置 Test Dataset Configuration
    test_name_pattern: str = "ai_challenger_pdr2018_test*.zip"  # 测试集ZIP文件的匹配模式
    use_all_test_datasets: bool = True  # 是否使用所有测试集
    primary_test_dataset: str = "testa"  # 如果不使用所有测试集，指定使用哪个测试集

    # 数据增强路径配置 Data Augmentation Path Configuration
    aug_source_path: str = field(default="")  # 数据增强源数据路径
    aug_target_path: str = field(default="")  # 数据增强结果保存路径

    # 设备配置 Device Configuration
    device: str = 'auto'  # 训练设备选择: "auto", "cuda", "cpu"
    gpus: str = '0'  # GPU设备ID: "0" 或 "0, 1"
    cuda_launch_blocking: bool = True  # 是否启用 CUDA_LAUNCH_BLOCKING
    cuda_alloc_conf: Optional[str] = "expandable_segments:True"  # PYTORCH_CUDA_ALLOC_CONF 配置

    # 训练参数 Training Parameters
    epoch: int = 40  # 训练轮数
    pretrained: bool = True  # 默认启用预训练权重以提升迁移学习效果
    train_batch_size: int = 8  # ConvNeXt V2 Base 384 默认使用更保守的批次大小
    val_batch_size: int = 8  # 验证批次大小
    test_batch_size: int = 8  # 测试批次大小
    num_workers: Union[int, str] = 32  # 数据加载线程数（可以是整数或 'auto'）
    img_height: int = 384  # 图像高度
    img_weight: int = 384  # 图像宽度
    num_classes: int = 59  # 类别数量
    seed: int = 888  # 随机种子
    lr: float = 3e-4  # 学习率
    lr_decay: float = 1e-4  # 学习率衰减
    weight_decay: float = 5e-2  # 权重衰减
    
    # 优化器配置 Optimizer Configuration
    optimizer: str = 'adamw'  # 优化器选择
    use_lookahead: bool = False  # 禁用Lookahead
    
    # 学习率调度配置 Learning Rate Scheduler Configuration
    scheduler: str = 'cosine'  # 使用余弦退火调度器
    warmup_epochs: int = 3  # 预热轮数
    warmup_factor: float = 0.1  # 预热因子
    
    # 数据增强参数 Data Augmentation Parameters
    use_mixup: bool = True  # 是否使用Mixup
    mixup_alpha: float = 0.4  # Mixup alpha参数
    cutmix_prob: float = 0.5  # CutMix概率
    use_random_erasing: bool = True  # 是否使用随机擦除
    use_data_aug: bool = True  # 是否使用数据增强
    use_weighted_sampler: bool = True  # 是否使用按类别频次加权的采样器
    weighted_sampler_power: float = 1.0  # 类别采样权重指数
    weighted_sampler_min_count: int = 1  # 采样时的最小类别计数下限
    
    # 训练策略参数 Training Strategy Parameters
    use_amp: bool = True  # 是否使用混合精度训练
    use_ema: bool = True  # 是否使用EMA
    ema_decay: float = 0.995  # EMA衰减率
    use_early_stopping: bool = True  # 是否使用早停
    early_stopping_patience: int = 10  # 早停耐心值
    gradient_clip_val: float = 1.0  # 梯度裁剪值
    use_gradient_checkpointing: bool = True  # 是否使用梯度检查点
    label_smoothing: float = 0.1  # 标签平滑系数
    use_focal_loss: bool = True  # 是否使用Focal Loss
    focal_loss_gamma: float = 2.0  # Focal Loss gamma参数
    focal_loss_alpha: float = 0.25  # Focal Loss alpha参数

    # 数据增强配置 Data Augmentation Configuration
    use_mode: str = 'merge'  # 数据增强模式: 'merge', 'replace'
    merge_augmented_data: bool = True  # 是否合并增强数据和原始数据
    aug_noise: bool = True  # 是否添加噪声
    aug_brightness: bool = True  # 是否调整亮度
    aug_flip: bool = True  # 是否进行翻转
    aug_contrast: bool = True  # 是否调整对比度
    remove_error_images: bool = True  # 是否删除错误图片
    aug_num_workers: int = 8  # 数据增强处理线程数

    # 数据增强处理参数 Data Augmentation Processing Parameters
    aug_max_workers: int = 8  # 数据增强最大线程数
    aug_noise_var: float = 0.02  # 高斯噪声方差
    aug_brightness_range: Tuple[float, float] = (0.3, 1.7)  # 亮度调整范围
    aug_contrast_factor: float = 2.0  # 对比度增强因子

    # 数据集采样配置 Dataset Sampling Configuration
    enable_sampling: bool = True  # 是否启用数据集采样
    sampling_threshold: int = 1000000  # 触发采样的数据集大小阈值
    sample_size: int = 50000  # 采样后的数据集大小
    min_files_threshold: int = 1000  # 最小文件数阈值
    enable_image_validation: bool = True  # 是否在加载数据集时进行图像验证
    image_validation_workers: int = 4  # 图像验证的最大线程数

    # 高级功能 Advanced Features
    progressive_resizing: bool = True  # 是否使用渐进式缩放
    progressive_start_size: int = 224  # 渐进式缩放起始尺寸
    progressive_end_size: int = 380  # 渐进式缩放最终尺寸
    progressive_epochs: int = 15  # 渐进式缩放过渡轮数
    progressive_sizes: List[int] = field(default_factory=lambda: [224, 320, 380])  # 渐进式缩放的图像尺寸
    tta_views: int = 4  # 推理阶段测试时增强视角数，支持 1/2/3/4

    def __post_init__(self):
        """初始化后的验证和设置"""
        # 初始化依赖于paths的字段
        self.train_data = self.paths.train_dir
        self.test_data = self.paths.test_images_dir
        self.weights = self.paths.weight_dir
        self.best_weights = self.paths.best_weight_dir
        self.submit = self.paths.submit_dir
        self.logs = self.paths.log_dir
        self.aug_source_path = self.paths.train_dir
        self.aug_target_path = self.paths.aug_train_dir

        # 规范化图像扩展名
        normalized_exts: List[str] = []
        for ext in self.image_extensions:
            if not ext:
                continue
            ext = ext.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext not in normalized_exts:
                normalized_exts.append(ext)
        self.image_extensions = tuple(normalized_exts)
        
        # 验证设备设置
        if self.device not in ['auto', 'cuda', 'cpu']:
            raise ValueError("device must be 'auto', 'cuda', or 'cpu'")

        # 设置工作线程数
        if self.num_workers == 'auto':
            try:
                import psutil  # type: ignore
                self.num_workers = psutil.cpu_count(logical=False)
            except ImportError:
                import multiprocessing
                self.num_workers = multiprocessing.cpu_count()
        elif isinstance(self.num_workers, str):
            # 如果传入的是字符串数字，转换为整数
            try:
                self.num_workers = int(self.num_workers)
            except ValueError:
                raise ValueError("num_workers must be 'auto', a positive integer, or a string representing an integer")
        elif not isinstance(self.num_workers, int) or self.num_workers <= 0:
            raise ValueError("num_workers must be 'auto' or a positive integer")

        # 设置数据增强线程数
        if self.aug_num_workers == 'auto':
            try:
                import psutil  # type: ignore
                self.aug_num_workers = psutil.cpu_count(logical=False)
            except ImportError:
                import multiprocessing
                self.aug_num_workers = multiprocessing.cpu_count()
        elif isinstance(self.aug_num_workers, str):
            # 如果传入的是字符串数字，转换为整数
            try:
                self.aug_num_workers = int(self.aug_num_workers)
            except ValueError:
                raise ValueError("aug_num_workers must be 'auto', a positive integer, or a string representing an integer")
        elif not isinstance(self.aug_num_workers, int) or self.aug_num_workers <= 0:
            raise ValueError("aug_num_workers must be 'auto' or a positive integer")

config = DefaultConfigs()
paths = config.paths
