import os
import io
import json
import time
import numpy as np
import torch
import logging
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.utils import MyEncoder, build_transforms, get_image_extensions, is_image_file
from config import config
from models.model import get_net

class InferenceManager:
    """植物病害检测推理管理器类"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_name: Optional[str] = None,
        logger=None,
        cfg=None,
    ):
        """初始化推理管理器
        
        参数:
            model_path: 模型权重路径
            device: 使用的设备('cuda', 'cpu'或None表示自动检测)
            model_name: 模型名称（覆盖默认配置）
            logger: 可选的日志记录器实例
        """
        self.config = cfg or config
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.model_name = model_name
        self.logger = logger or self._setup_logger()
        
    def _setup_logger(self):
        """设置并返回推理日志记录器"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.paths.inference_log),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('Inference')
    
    def _get_device(self, device_str: Optional[str] = None) -> torch.device:
        """获取适合推理的设备
        
        参数:
            device_str: 设备指定('cuda', 'cpu'或None表示自动)
            
        返回:
            torch.device对象
        """
        if device_str == "cuda":
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif device_str == "cpu":
            return torch.device('cpu')
        else:
            # 自动模式
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def load_model(self, model_path: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """从检查点加载模型
        
        参数:
            model_path: 模型权重路径(如果为None则使用self.model_path)
            model_name: 模型名称(如果为None则使用self.model_name或config.model_name)
        """
        model_path = model_path or self.model_path
        if model_path is None:
            raise ValueError("No model path provided")
        model_name = model_name or self.model_name or self.config.model_name
            
        self.logger.info(f"Loading model from {model_path}")
        
        # 初始化模型架构
        model = get_net(
            model_name=model_name,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
        )
        
        # 加载权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的检查点格式
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        # 将模型移至设备并设置为评估模式
        self.model = model.to(self.device)
        self.model.eval()
        
    def predict_single(self, image_path: str) -> np.ndarray:
        """对单张图像进行预测
        
        参数:
            image_path: 图像文件路径
            
        返回:
            类别概率的NumPy数组
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first")
            
        # 准备图像变换
        transform = build_transforms(train=False, test=True, cfg=self.config)
        
        # 加载并变换图像
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
            
        # 进行预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.Softmax(dim=1)(output)
            
        return probabilities.cpu().numpy()[0]
    
    def predict_batch(
        self,
        image_folder: str,
        batch_size: int = 16,
        num_workers: int = 4,
        topk: int = 3,
        return_probabilities: bool = False,
        return_topk: bool = True,
        confidence_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """对文件夹中的所有图像进行预测
        
        参数:
            image_folder: 包含图像的文件夹路径
            batch_size: 推理的批次大小
            num_workers: 数据加载的工作线程数
            topk: 输出Top-K预测结果
            return_probabilities: 是否返回完整概率向量
            return_topk: 是否返回Top-K列表
            confidence_threshold: 低于阈值则标记为low_confidence
            
        返回:
            包含预测结果的字典列表
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first")
            
        # Get image files
        if not os.path.exists(image_folder) or not os.path.isdir(image_folder):
            abs_folder = os.path.abspath(image_folder)
            self.logger.error(f"Image folder not found or not a directory: {abs_folder}")
            return []

        candidates = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        image_extensions = get_image_extensions(cfg=self.config)
        image_files = [
            path for path in candidates
            if os.path.isfile(path) and is_image_file(path, image_extensions)
        ]
        image_files.sort()
        
        if not image_files:
            self.logger.warning(f"No images found in {image_folder}")
            return []

        skipped = len(candidates) - len(image_files)
        if skipped > 0:
            self.logger.info(f"Skipped {skipped} non-image entries in {image_folder}")
            
        self.logger.info(f"Found {len(image_files)} image files")
        
        # Create dataset and dataloader
        dataset = InferenceDataset(image_files, cfg=self.config)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        # Make predictions
        results = []
        
        with torch.no_grad():
            for batch_images, batch_paths in tqdm(dataloader, desc="Making predictions"):
                batch_images = batch_images.to(self.device)
                outputs = self.model(batch_images)
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.Softmax(dim=1)(outputs)
                safe_topk = max(1, min(topk, probabilities.size(1)))
                topk_scores, topk_indices = torch.topk(probabilities, k=safe_topk, dim=1)
                
                # Process predictions
                for i, (probs, path) in enumerate(zip(probabilities, batch_paths)):
                    pred_index = int(torch.argmax(probs).item())
                    pred_label = self._remap_label_index(pred_index)
                    confidence = float(torch.max(probs).item())

                    # Create result entry
                    result: Dict[str, Any] = {
                        "image_id": os.path.basename(path),
                        "disease_class": pred_label,
                        "confidence": confidence,
                    }

                    if return_topk:
                        topk_items = []
                        for score, idx in zip(topk_scores[i], topk_indices[i]):
                            topk_items.append({
                                "class": self._remap_label_index(int(idx.item())),
                                "score": float(score.item()),
                            })
                        result["topk"] = topk_items

                    if return_probabilities:
                        result["probabilities"] = probs.cpu().numpy().tolist()

                    if confidence_threshold is not None:
                        result["low_confidence"] = confidence < confidence_threshold

                    results.append(result)
        
        self.logger.info(f"Processed {len(results)} images")
        return results
    
    def save_predictions(
        self,
        predictions: List[Dict],
        output_file: Optional[str] = None,
        output_format: str = "submit",
    ) -> None:
        """将预测结果保存到JSON文件
        
        参数:
            predictions: 预测字典列表
            output_file: 输出文件路径
            output_format: submit(提交格式)或full(完整输出)
        """
        output_file = output_file or self.config.paths.prediction_file
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Prepare output format
        if output_format == "submit":
            output_payload = [
                {
                    "image_id": pred["image_id"],
                    "disease_class": pred["disease_class"],
                }
                for pred in predictions
            ]
        elif output_format == "full":
            output_payload = predictions
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Save to file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, ensure_ascii=False, cls=MyEncoder)
            self.logger.info(f"Predictions saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            raise

    @staticmethod
    def _remap_label_index(pred_label: int) -> int:
        if pred_label > 43:
            return pred_label + 2
        return pred_label

class InferenceDataset(Dataset):
    """推理数据集类"""
    
    def __init__(self, file_paths, cfg=None):
        """初始化数据集
        
        参数:
            file_paths: 图像文件路径列表
        """
        self.config = cfg or config
        self.file_paths = file_paths
        self.transforms = build_transforms(train=False, test=True, cfg=self.config)
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        """获取单个数据样本
        
        参数:
            idx: 索引
            
        返回:
            (图像张量, 图像路径)元组
        """
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transforms(image)
            return image, img_path
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            # Return empty image
            return torch.zeros((3, self.config.img_height, self.config.img_weight)), img_path

def predict(
    model_path: str,
    input_path: str,
    output_file: Optional[str] = None,
    is_dir: bool = False,
    device: Optional[str] = None,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    topk: int = 3,
    save_probs: bool = False,
    output_format: str = "submit",
    confidence_threshold: Optional[float] = None,
    cfg=None,
) -> List[Dict]:
    """预测函数，支持单张图片或图片目录
    
    参数:
        model_path: 模型权重路径
        input_path: 输入图片或图片目录路径
        output_file: 输出文件路径
        is_dir: 输入是否为目录
        device: 推理设备
        batch_size: 批次大小（目录推理）
        num_workers: 数据加载线程数
        topk: 输出Top-K预测
        save_probs: 是否保存完整概率向量
        output_format: 输出格式 submit/full
        confidence_threshold: 低于阈值则标记low_confidence
        
    返回:
        预测结果字典列表
    """
    logger = logging.getLogger('Inference')
    cfg = cfg or config
    output_file = output_file or cfg.paths.prediction_file
    
    # 初始化推理管理器
    inference = InferenceManager(model_path, device=device, model_name=model_name, cfg=cfg)
    
    try:
        # 加载模型
        inference.load_model(model_name=model_name)
        
        # 进行预测
        if is_dir:
            # 检查目录是否存在
            if not os.path.exists(input_path) or not os.path.isdir(input_path):
                abs_input = os.path.abspath(input_path)
                logger.error(f"Directory not found or not a directory: {abs_input}")
                return []
                
            # 检查目录是否为空
            image_extensions = get_image_extensions(cfg=cfg)
            candidates = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
            ]
            image_files = [
                path
                for path in candidates
                if os.path.isfile(path) and is_image_file(path, image_extensions)
            ]
            image_files.sort()
            if not image_files:
                logger.error(f"No image files found in directory: {input_path}")
                return []
                
            # 目录预测
            logger.info(f"Predicting on {len(image_files)} images in {input_path}")
            predictions = inference.predict_batch(
                input_path,
                batch_size=batch_size or cfg.test_batch_size,
                num_workers=num_workers or cfg.num_workers,
                topk=topk,
                return_probabilities=save_probs,
                return_topk=True,
                confidence_threshold=confidence_threshold,
            )
        else:
            # 单张图片预测
            if not os.path.exists(input_path) or not os.path.isfile(input_path):
                abs_input = os.path.abspath(input_path)
                logger.error(f"File not found or not a file: {abs_input}")
                return []
                
            # 对单个图像进行预测并将结果转换为列表格式
            logger.info(f"Predicting on single image: {input_path}")
            pred = inference.predict_single(input_path)
            topk_safe = max(1, min(topk, len(pred)))
            topk_indices = np.argsort(pred)[::-1][:topk_safe]
            topk_items = [
                {
                    "class": inference._remap_label_index(int(idx)),
                    "score": float(pred[idx]),
                }
                for idx in topk_indices
            ]

            result = {
                'image_id': os.path.basename(input_path),
                'disease_class': inference._remap_label_index(int(np.argmax(pred))),
                'confidence': float(np.max(pred)),
                'topk': topk_items,
            }
            if save_probs:
                result['probabilities'] = pred.tolist()
            if confidence_threshold is not None:
                result["low_confidence"] = float(np.max(pred)) < confidence_threshold
            predictions = [result]
        
        # 保存预测结果
        if output_file and predictions:
            inference.save_predictions(predictions, output_file, output_format=output_format)
            logger.info(f"Saved predictions to {output_file}")
            
        return predictions
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [] 
