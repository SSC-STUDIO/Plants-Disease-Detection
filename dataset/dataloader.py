import random 
import numpy as np 
import pandas as pd 
import torch 
import os
import logging
from itertools import chain 
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from config import config, paths
from PIL import Image 
from concurrent.futures import ThreadPoolExecutor
from utils.utils import handle_datasets, build_transforms, get_image_glob_patterns, get_image_extensions
import concurrent.futures

# SECURITY FIX: Import security modules
from libs.image_security import (
    SecureImageLoader,
    ImageSecurityError,
    ImageTooLargeError,
    ImageValidationError,
    secure_load_image
)
from libs.data_validation import (
    DataSanitizer,
    DataValidationResult,
    SecureDatasetLoader,
    validate_data_path
)

# 最大图像尺寸限制 (100MP, 约400MB内存)
MAX_IMAGE_SIZE = 100_000_000

# 设置日志记录器
logger = logging.getLogger('DataLoader')
logger.setLevel(logging.INFO)

# 创建日志处理器
if not logger.handlers:
    # 文件处理器
    if not os.path.exists(paths.log_dir):
        os.makedirs(paths.log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(paths.log_dir, 'dataloader.log'))
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# SECURITY FIX: Configure PIL to prevent decompression bomb attacks
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_SIZE

def seed_everything(cfg) -> None:
    """设置随机种子，便于复现实验。"""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

class PlantDiseaseDataset(Dataset):
    """植物病害图像数据集类 - 安全增强版本"""
    def __init__(
        self,
        label_list,
        sampling_threshold,
        sample_size=None,
        seed=None,
        img_width=None,
        img_height=None,
        use_data_aug=None,
        transforms=None,
        train=True,
        test=False,
        enable_sampling=None,
        validate_images=None,
        validation_workers=None,
        cfg=None,
        seed_all: bool = True,
        strict_validation: bool = True,  # SECURITY FIX: Enable strict validation by default
    ):
        """初始化数据集
        
        参数:
            label_list: 包含文件路径和标签的DataFrame
            transforms: 数据增强转换
            train: 是否为训练模式
            test: 是否为测试模式
            strict_validation: 是否启用严格的安全验证
        """
        self.config = cfg or config
        if seed_all:
            seed_everything(self.config)

        self.test = test 
        self.train = train 
        self.enable_sampling = self.config.enable_sampling if enable_sampling is None else enable_sampling
        self.sampling_threshold = sampling_threshold
        self.sample_size = self.config.sample_size if sample_size is None else sample_size
        self.seed = self.config.seed if seed is None else seed
        self.img_width = self.config.img_width if img_width is None else img_width
        self.img_height = self.config.img_height if img_height is None else img_height
        self.use_data_aug = self.config.use_data_aug if use_data_aug is None else use_data_aug
        self.transforms = self._get_transforms(transforms, train, test)
        self.validate_images = self.config.enable_image_validation if validate_images is None else validate_images
        self.validation_workers = validation_workers if validation_workers is not None else self.config.image_validation_workers
        
        # SECURITY FIX: Initialize security components
        self.strict_validation = strict_validation
        self.secure_image_loader = SecureImageLoader(
            max_pixels=getattr(self.config, 'safe_max_image_pixels', MAX_IMAGE_SIZE),
            max_dimension=getattr(self.config, 'safe_max_image_dimension', 10000),
            max_file_size=getattr(self.config, 'safe_max_file_size', 100 * 1024 * 1024),
            min_file_size=getattr(self.config, 'safe_min_file_size', 100)
        )
        self.data_sanitizer = DataSanitizer(base_path=self.config.paths.data_dir if hasattr(self.config, 'paths') else None)
        
        self.imgs = self._load_images(label_list)
        
    def _load_images(self, label_list):
        """加载并验证图像 - 安全增强版本
        
        参数:
            label_list: 包含文件路径和标签的DataFrame
            
        返回:
            有效的图像数据列表
        """
        if self.test:
            # SECURITY FIX: Validate test image paths
            valid_files = []
            for _, row in label_list.iterrows():
                filename = row["filename"]
                result = self.data_sanitizer.validate_file_path(filename, must_exist=True)
                if result.is_valid:
                    valid_files.append(result.sanitized_data)
                else:
                    logger.warning(f"Skipping invalid test file path: {filename}, errors: {result.errors}")
            return valid_files
        
        # 将DataFrame转换为列表以提高处理速度
        imgs = list(zip(label_list["filename"], label_list["label"]))
        
        # SECURITY FIX: Validate and sanitize all file paths first
        sanitized_imgs = []
        for filename, label in imgs:
            result = self.data_sanitizer.sanitize_dataset_record(
                {"filename": filename, "label": label},
                allowed_extensions={'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            )
            if result.is_valid:
                sanitized_imgs.append((result.sanitized_data["filename"], result.sanitized_data["label"]))
            else:
                logger.warning(f"Skipping invalid record: {filename}, errors: {result.errors}")
        
        imgs = sanitized_imgs
        
        # 对于大数据集，打印警告并建议采样
        if len(imgs) > 50000:
            print(f"Warning: The dataset is very large ({len(imgs)} images), it may cause memory problems")
            print("Hint: Consider using smaller batch size or reducing the dataset size.")

        if not self.validate_images:
            logger.info("Image validation is disabled; skipping integrity checks")
            if self.enable_sampling and len(imgs) > self.sampling_threshold:
                print(f"\nThe dataset is very large ({len(imgs)} images), Conduct sampling to reduce memory usage")
                random.seed(self.seed)
                imgs = random.sample(imgs, self.sample_size)
                print(f"The size of the sampled dataset: {len(imgs)} images")
            return imgs

        valid_imgs = []
        invalid_imgs = []
        
        def validate_image_batch_secure(batch):
            """安全验证一批图像
            
            参数:
                batch: 图像数据(文件名, 标签)元组的列表
                
            返回:
                (有效图像列表, 无效图像列表)元组
            """
            valid = []
            invalid = []
            
            for img_data in batch:
                try:
                    filename = img_data[0]
                    
                    # SECURITY FIX: Use secure image loader for validation
                    # First do quick file checks
                    if not os.path.exists(filename):
                        invalid.append(img_data)
                        continue
                    
                    # SECURITY FIX: Validate file size
                    try:
                        file_size = os.path.getsize(filename)
                        if file_size < self.secure_image_loader.min_file_size:
                            logger.warning(f"File too small, possible corruption: {filename} ({file_size} bytes)")
                            invalid.append(img_data)
                            continue
                        if file_size > self.secure_image_loader.max_file_size:
                            logger.warning(f"File too large, possible attack: {filename} ({file_size} bytes)")
                            invalid.append(img_data)
                            continue
                    except OSError:
                        invalid.append(img_data)
                        continue
                    
                    # SECURITY FIX: Validate image integrity without loading full image
                    is_valid, error_msg = self.secure_image_loader.verify_image_integrity(filename)
                    if not is_valid:
                        logger.warning(f"Image integrity check failed for {filename}: {error_msg}")
                        invalid.append(img_data)
                        continue
                    
                    # Additional check: try to get image size without loading pixels
                    try:
                        with Image.open(filename) as img:
                            width, height = img.size
                            if width * height > self.secure_image_loader.max_pixels:
                                logger.warning(f"Image too large (decompression bomb?): {filename} ({width}x{height})")
                                invalid.append(img_data)
                                continue
                            if width > self.secure_image_loader.max_dimension or height > self.secure_image_loader.max_dimension:
                                logger.warning(f"Image dimensions too large: {filename} ({width}x{height})")
                                invalid.append(img_data)
                                continue
                    except Exception as e:
                        logger.warning(f"Cannot read image metadata: {filename}: {e}")
                        invalid.append(img_data)
                        continue
                    
                    valid.append(img_data)
                except Exception as e:
                    logger.warning(f"Unexpected error validating {img_data[0]}: {e}")
                    invalid.append(img_data)
            
            return valid, invalid
        
        # 使用多线程并行验证图像
        print("Validating images with security checks...")
        
        # 更高效的批处理配置
        batch_size = 100  # 更大的批次以减少线程创建开销
        cpu_count = os.cpu_count() or 1
        max_workers = min(max(1, int(self.validation_workers)), cpu_count)
        
        # 将图像分成批次
        batches = []
        for i in range(0, len(imgs), batch_size):
            batches.append(imgs[i:min(i + batch_size, len(imgs))])
        
        # 使用线程池并行处理所有批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 使用tqdm显示进度
            futures = []
            for batch in batches:
                futures.append(executor.submit(validate_image_batch_secure, batch))
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc="Validating images"):
                try:
                    batch_valid, batch_invalid = future.result()
                    valid_imgs.extend(batch_valid)
                    invalid_imgs.extend(batch_invalid)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
        
        # 对于极大的数据集，进行采样以减少内存压力
        if self.enable_sampling and len(valid_imgs) > self.sampling_threshold:
            print(f"\nThe dataset is very large ({len(valid_imgs)} effective images), Conduct sampling to reduce memory usage")
            import random
            random.seed(self.seed)  # 保持一致性
            valid_imgs = random.sample(valid_imgs, self.sample_size)
            print(f"The size of the sampled dataset: {len(valid_imgs)} images")
        
        if invalid_imgs:
            print(f"\nFound {len(invalid_imgs)} invalid images that will be skipped:")
            for _, (filename, _) in enumerate(invalid_imgs[:5], 1):
                print(f"  {filename}")
            if len(invalid_imgs) > 5:
                print(f"  ... and {len(invalid_imgs) - 5} more")
            
        print(f"Successfully loaded {len(valid_imgs)} valid images")
        
        # 强制清理内存
        import gc
        gc.collect()
        
        return valid_imgs
        
    def _get_transforms(self, transforms, train, test):
        """获取数据转换操作
        
        参数:
            transforms: 自定义转换
            train: 是否为训练模式
            test: 是否为测试模式
            
        返回:
            转换操作序列
        """
        if transforms is not None:
            return transforms
            
        return build_transforms(
            train=train,
            test=test,
            use_data_aug=self.use_data_aug,
            img_height=self.img_height,
            img_width=self.img_width,
            cfg=self.config,
        )

    def __getitem__(self, index):
        """获取单个数据样本 - 安全增强版本
        
        参数:
            index: 索引
            
        返回:
            (图像张量, 标签)或(图像张量, 文件名)
        """
        try:
            if self.test:
                filename = self.imgs[index]
                # SECURITY FIX: Use secure image loading
                img = self.secure_image_loader.load_image(filename, mode='RGB')
                img_tensor = self.transforms(img)
                return img_tensor, filename
            else:
                filename, label = self.imgs[index]
                # SECURITY FIX: Use secure image loading
                img = self.secure_image_loader.load_image(filename, mode='RGB')
                img_tensor = self.transforms(img)
                return img_tensor, label
        except ImageSecurityError as e:
            logger.warning(f"Image security error at index {index}: {str(e)}")
            # 返回空张量作为错误处理
            if self.test:
                return (torch.zeros((3, self.img_height, self.img_width)), 
                       self.imgs[index] if index < len(self.imgs) else "")
            else:
                return (torch.zeros((3, self.img_height, self.img_width)), 
                       self.imgs[index][1] if index < len(self.imgs) else 0)
        except Exception as e:
            logger.error(f"Error loading image at index {index}: {str(e)}")
            # 返回空张量作为错误处理
            if self.test:
                return (torch.zeros((3, self.img_height, self.img_width)), 
                       self.imgs[index] if index < len(self.imgs) else "")
            else:
                return (torch.zeros((3, self.img_height, self.img_width)), 
                       self.imgs[index][1] if index < len(self.imgs) else 0)
                
    def __len__(self):
        """返回数据集大小"""
        return len(self.imgs)

def collate_fn(batch):
    """批次数据收集函数

def get_files(data_path, mode, cfg=None):
    """获取数据集文件路径和标签 - 安全增强版本
    
    参数:
        data_path: 数据集根目录路径
        mode: 'train', 'val' 或 'test' 模式
        
    返回:
        包含文件路径和标签的DataFrame
    """
    # 使用传入的数据路径，而不是调用handle_datasets
    actual_root = data_path
    
    if not os.path.exists(actual_root):
        raise FileNotFoundError(f"Directory not found: {actual_root}")
    
    logger.info(f"Loading {mode} dataset from: {actual_root}")
    
    cfg = cfg or config
    
    # SECURITY FIX: Initialize data sanitizer for path validation
    data_sanitizer = DataSanitizer(base_path=cfg.paths.data_dir if hasattr(cfg, 'paths') else None)

    if mode == "test":
        image_exts = get_image_extensions(cfg=cfg)
        files = []
        for img in os.listdir(actual_root):
            img_path = os.path.join(actual_root, img)
            # SECURITY FIX: Validate file path
            result = data_sanitizer.validate_file_path(
                img_path, 
                must_exist=True,
                allowed_extensions=set(ext.lower() for ext in image_exts)
            )
            if result.is_valid:
                files.append(result.sanitized_data)
            else:
                logger.warning(f"Skipping invalid test file: {img_path}")
        files.sort()
        return pd.DataFrame({"filename": files})
        
    elif mode in ["train", "val"]: 
        all_data_path, labels = [], []
        
        # SECURITY FIX: Validate directory listing
        try:
            dir_contents = os.listdir(actual_root)
        except OSError as e:
            raise PermissionError(f"Cannot access directory {actual_root}: {e}")
        
        image_folders = []
        for x in dir_contents:
            folder_path = os.path.join(actual_root, x)
            # SECURITY FIX: Validate folder path
            result = data_sanitizer.validate_file_path(folder_path, must_exist=True)
            if result.is_valid and os.path.isdir(folder_path):
                image_folders.append(result.sanitized_data)
        
        # 获取所有jpg和png图像路径
        image_patterns = [f"/{pattern}" for pattern in get_image_glob_patterns(cfg=cfg)]
        all_images = []
        for folder in image_folders:
            for pattern in image_patterns:
                # SECURITY FIX: Sanitize glob results
                matched_files = glob(folder + pattern)
                for file_path in matched_files:
                    result = data_sanitizer.validate_file_path(file_path, must_exist=True)
                    if result.is_valid:
                        all_images.append(result.sanitized_data)
                    else:
                        logger.warning(f"Skipping invalid image path: {file_path}")
        all_images.sort()
                
        logger.info(f"Loading {mode} dataset ({len(all_images)} images)")
        
        for file in tqdm(all_images):
            # SECURITY FIX: Validate label extraction
            try:
                label_str = os.path.basename(os.path.dirname(file))
                # Validate label is a valid integer
                label = int(label_str)
                if label < 0:
                    logger.warning(f"Negative label found for {file}: {label}")
                    continue
                all_data_path.append(file)
                labels.append(label)
            except ValueError as e:
                logger.warning(f"Invalid label for {file}: {e}")
                continue
            
        return pd.DataFrame({
            "filename": all_data_path,
            "label": labels
        })
        
    else:
        raise ValueError("Mode must be one of 'train', 'val', or 'test'")
