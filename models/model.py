import torch
import torchvision
import torch.nn.functional as F 
from torch import nn
import os

try:
    import timm
except ImportError:
    timm = None


def _build_torchvision_model(builder, weights_attr, pretrained):
    weights_enum = getattr(torchvision.models, weights_attr, None)
    if weights_enum is not None:
        return builder(weights=weights_enum.DEFAULT if pretrained else None)
    return builder(pretrained=pretrained)


def _freeze_all_parameters(model):
    """冻结模型所有参数"""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_all_parameters(model):
    """解冻模型所有参数（用于从头训练或无预训练权重时）"""
    for param in model.parameters():
        param.requires_grad = True


def _unfreeze_matching_parameters(model, trainable_keywords):
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True


def _ensure_pretrained_cache_dir():
    """确保 TORCH_HOME/HF_HOME 指向项目内缓存目录且目录存在。

    设置环境变量可避免预训练权重下载到系统默认目录，同时确保目标
    路径可写入。在设置成功时不覆盖已有的环境变量，保留用户自定义路径。
    """
    _pretrained_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'pretrained')
    if os.environ.get('TORCH_HOME') or os.environ.get('HF_HOME'):
        # 用户已通过环境变量覆盖，不强行设置
        return
    try:
        os.makedirs(_pretrained_dir, exist_ok=True)
        os.environ['TORCH_HOME'] = _pretrained_dir
        os.environ['HF_HOME'] = _pretrained_dir
    except OSError as exc:
        print(f"Warning: could not create pretrained cache dir {_pretrained_dir}: {exc}")


def _create_timm_model_with_retry(model_names, pretrained, **kwargs):
    """创建 timm 模型，支持重试和预训练权重回退。

    Args:
        model_names: 模型名称候选列表
        pretrained: 是否使用预训练权重
        **kwargs: 传递给 timm.create_model 的其他参数

    Returns:
        元组 (model, pretrained_loaded):
            - model: 创建的模型
            - pretrained_loaded: 是否成功加载了预训练权重
    """
    last_error = None

    # 如果请求预训练权重，先尝试加载
    if pretrained:
        for model_name in model_names:
            try:
                model = timm.create_model(model_name, pretrained=True, **kwargs)
                print(f"Created model with pretrained weights: {model_name}")
                return model, True  # 成功加载预训练权重
            except Exception as exc:
                last_error = exc
                print(f"Failed to load pretrained weights for {model_name}: {exc}")

        # 预训练权重下载失败，回退到随机初始化
        print("Falling back to randomly initialized weights after pretrained download failure")

    # 尝试随机初始化
    for model_name in model_names:
        try:
            model = timm.create_model(model_name, pretrained=False, **kwargs)
            print(f"Created model without pretrained weights: {model_name}")
            return model, False  # 使用随机初始化
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to create timm model from candidates {model_names}: {last_error}") from last_error

def get_densenet169(num_classes, pretrained=True):
    """生成DenseNet169模型
    Args:
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    class DenseModel(nn.Module):
        def __init__(self, pretrained_model):
            super(DenseModel, self).__init__()
            self.features = pretrained_model.features
            self.classifier = nn.Sequential(
                nn.Linear(pretrained_model.classifier.in_features, num_classes)
            )
            
            self._initialize_weights()
            
        def _initialize_weights(self):
            """初始化模型权重"""
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            """前向传播函数"""
            features = self.features(x)
            out = F.adaptive_avg_pool2d(features, (1, 1))
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out

    return DenseModel(_build_torchvision_model(
        torchvision.models.densenet169, "DenseNet169_Weights", pretrained))

def get_efficientnet(num_classes, pretrained=True):
    """获取EfficientNet-B4模型，使用渐进式解冻技术
    Args:
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    model = _build_torchvision_model(
        torchvision.models.efficientnet_b4,
        "EfficientNet_B4_Weights",
        pretrained,
    )
    
    # 冻结大部分层
    for name, param in model.named_parameters():
        if 'features.8' not in name:  # 只训练最后几层
            param.requires_grad = False
    
    # 替换分类器
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def get_efficientnetv2(num_classes, pretrained=True):
    """获取EfficientNetV2-S模型，性能更好

    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重，默认为True

    Returns:
        EfficientNetV2-S 模型
    """
    # SSL验证保持默认启用状态以确保安全
    # 如需使用特定证书，可通过环境变量 SSL_CERT_FILE 配置

    _ensure_pretrained_cache_dir()

    if timm is None:
        print("timm is not installed, falling back to torchvision efficientnet_v2_s")
        model = _build_torchvision_model(
            torchvision.models.efficientnet_v2_s,
            "EfficientNet_V2_S_Weights",
            pretrained,
        )
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model

    pretrained_loaded = False
    model = None

    # 尝试加载预训练权重
    if pretrained:
        try:
            model = timm.create_model(
                'tf_efficientnetv2_s',
                pretrained=True,
                num_classes=0,
                features_only=False,
                out_indices=None,
            )
            pretrained_loaded = True
            print("Created EfficientNetV2-S model with pretrained weights")
        except Exception as e:
            print(f"Failed to load pretrained EfficientNetV2-S: {e}, falling back to random init")

    # 如果没有成功加载预训练权重，使用随机初始化
    if model is None:
        try:
            model = timm.create_model(
                'tf_efficientnetv2_s',
                pretrained=False,
                num_classes=0,
                features_only=False,
                out_indices=None,
            )
            print("Created EfficientNetV2-S model without pretrained weights")
        except Exception as e:
            raise RuntimeError(f"Failed to create EfficientNetV2-S model: {str(e)}") from e

    # 获取特征维度
    feature_dim = model.num_features

    # 修改分类头
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes)
    )

    # 只有成功加载预训练权重时才使用渐进式解冻
    if pretrained_loaded:
        # 使用渐进式解冻
        for param in model.parameters():
            param.requires_grad = False

        # 只训练最后几层
        trainable_layers = [
            model.classifier,
            model.blocks[-1],
            model.blocks[-2]
        ]

        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

        print("EfficientNetV2-S: partial freeze for fine-tuning")
    else:
        # 随机初始化，训练所有层
        _unfreeze_all_parameters(model)
        print("EfficientNetV2-S: full training mode (no pretrained weights)")

    return model

def get_convnext(num_classes, pretrained=True):
    """获取ConvNeXt模型
    Args:
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    model = _build_torchvision_model(
        torchvision.models.convnext_small,
        "ConvNeXt_Small_Weights",
        pretrained,
    )
    
    # 冻结早期层
    for name, param in model.named_parameters():
        if 'features.7' not in name:  # 只训练最后一个阶段
            param.requires_grad = False
    
    # 替换分类器
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model


def get_eva02_base(num_classes, pretrained=True):
    """获取EVA-02 Base模型 - 2024年最先进的视觉Transformer（针对8GB显存优化）

    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重，默认为True

    Returns:
        EVA-02 Base模型
    """
    if timm is None:
        raise ImportError("timm is required for EVA-02 model. Install with: pip install timm>=0.9.0")

    # EVA-02模型名称候选列表（Base版本更适合8GB显存）
    model_names = [
        "eva02_base_patch14_224.mim_in22k",  # Base 224px版本 - 正确的预训练标签
        "eva02_small_patch14_224",  # Small版本作为备选（无预训练标签）
    ]

    # 尝试创建模型，支持重试机制
    model, pretrained_loaded = _create_timm_model_with_retry(
        model_names,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=0.05,  # 较低的drop path率
    )

    # 只有成功加载预训练权重时才使用渐进式解冻策略
    # 如果是随机初始化，则训练所有层
    if pretrained_loaded:
        _freeze_all_parameters(model)
        _unfreeze_matching_parameters(model, ("blocks.11", "blocks.10", "norm", "head"))
        print("EVA-02 Base model created with pretrained weights (partial freeze for fine-tuning)")
    else:
        _unfreeze_all_parameters(model)
        print("EVA-02 Base model created without pretrained weights (full training mode)")

    return model


# 向后兼容别名：旧配置中可能引用了 "eva02_large"
get_eva02_large = get_eva02_base


def get_convnextv2_base_384(num_classes, pretrained=True):
    """获取更现代的 ConvNeXt V2 Base 384 模型。

    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重，默认为True

    Returns:
        ConvNeXt V2 Base 384 模型
    """
    if timm is None:
        print("timm is not installed, falling back to torchvision convnext_small")
        return get_convnext(num_classes, pretrained=pretrained)

    model, pretrained_loaded = _create_timm_model_with_retry(
        [
            "convnextv2_base.fcmae_ft_in22k_in1k_384",
            "convnextv2_base.fcmae_ft_in22k_in1k",
        ],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=0.2,
    )

    # 只有成功加载预训练权重时才使用渐进式解冻策略
    if pretrained_loaded:
        _freeze_all_parameters(model)
        _unfreeze_matching_parameters(model, ("stages.3", "norm", "head"))
        print("ConvNeXt V2 Base 384 model created with pretrained weights (partial freeze for fine-tuning)")
    else:
        _unfreeze_all_parameters(model)
        print("ConvNeXt V2 Base 384 model created without pretrained weights (full training mode)")

    return model

def get_swin_transformer(num_classes, pretrained=True):
    """获取Swin Transformer模型，适用于细粒度分类任务

    使用与 EVA-02 / ConvNeXt V2 相同的重试机制和缓存目录设置，
    确保预训练权重下载失败时优雅回退到随机初始化。

    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重，默认为True
    """
    if timm is None:
        raise ImportError("timm is required for swin_transformer model")

    _ensure_pretrained_cache_dir()

    model, pretrained_loaded = _create_timm_model_with_retry(
        ["swin_small_patch4_window7_224"],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=0.2,
    )

    # 只有成功加载预训练权重时才使用渐进式解冻
    if pretrained_loaded:
        _freeze_all_parameters(model)
        # 只解冻最后一个 stage 和分类头
        _unfreeze_matching_parameters(model, ("layers.3", "norm", "head"))
        print("Swin Transformer model created with pretrained weights (partial freeze for fine-tuning)")
    else:
        _unfreeze_all_parameters(model)
        print("Swin Transformer model created without pretrained weights (full training mode)")

    return model

def get_hybrid_model(num_classes, pretrained=True):
    """创建混合模型（CNN+Transformer），结合卷积网络的局部特征和Transformer的全局特征
    Args:
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    if timm is None:
        raise ImportError("timm is required for hybrid_model")

    class HybridModel(nn.Module):
        def __init__(self):
            super(HybridModel, self).__init__()
            # CNN部分: 使用EfficientNet提取特征
            self.cnn_model = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)
            
            # 冻结CNN早期层
            for name, param in self.cnn_model.named_parameters():
                if 'blocks.5' not in name and 'blocks.6' not in name:  # 只训练最后两个块
                    param.requires_grad = False
            
            # Transformer部分: 使用轻量级Transformer
            cnn_channels = self.cnn_model.feature_info.channels()[-1]  # 获取最后一层特征图的通道数
            self.transformer = timm.create_model(
                'vit_small_patch16_224', 
                pretrained=pretrained,
                img_size=16,  # 特征图大小
                patch_size=1,  # 使用1x1的patch
                in_chans=cnn_channels,  # 输入通道数为CNN输出的通道数
                num_classes=0  # 不使用分类头
            )
            
            # 冻结Transformer早期层
            for name, param in self.transformer.named_parameters():
                if 'blocks.10' not in name and 'blocks.11' not in name and 'norm' not in name:
                    param.requires_grad = False
            
            transformer_dim = self.transformer.embed_dim
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.LayerNorm(transformer_dim),
                nn.Dropout(0.3),
                nn.Linear(transformer_dim, num_classes)
            )
        
        def forward(self, x):
            # 通过CNN提取特征
            features = self.cnn_model(x)[-1]  # 获取最后一层特征图
            
            # 通过Transformer处理特征
            transformer_out = self.transformer.forward_features(features)
            
            # 分类
            out = self.classifier(transformer_out)
            return out
    
    return HybridModel()

def get_ensemble_model(num_classes, pretrained=True):
    """创建模型集成，结合多个模型的优势
    Args:
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    class EnsembleModel(nn.Module):
        def __init__(self):
            super(EnsembleModel, self).__init__()
            
            # 加载多个预训练模型
            self.model1 = get_efficientnetv2(num_classes=num_classes, pretrained=pretrained)
            self.model2 = get_convnext(num_classes=num_classes, pretrained=pretrained)
            
            # 确保这些模型的输出层是一致的
            # 集成层 - 使用注意力机制进行加权
            self.attention = nn.Sequential(
                nn.Linear(num_classes * 2, 2),
                nn.Softmax(dim=1)
            )
            
        def forward(self, x):
            # 获取各个模型的输出
            out1 = self.model1(x)
            out2 = self.model2(x)
            
            # 拼接输出
            combined = torch.cat((out1, out2), dim=1)
            
            # 计算注意力权重
            weights = self.attention(combined)
            
            # 加权合并
            out = weights[:, 0:1] * out1 + weights[:, 1:2] * out2
            
            return out
    
    return EnsembleModel()

MODEL_REGISTRY = {
    "densenet169": get_densenet169,
    "efficientnet_b4": get_efficientnet,
    "efficientnetv2_s": get_efficientnetv2,
    "convnext_small": get_convnext,
    "convnextv2_base_384": get_convnextv2_base_384,
    "eva02_base": get_eva02_base,  # EVA-02 Base（2024年最新架构）
    "eva02_large": get_eva02_base,  # 向后兼容别名，实际加载EVA-02 Base
    "swin_transformer": get_swin_transformer,
    "hybrid_model": get_hybrid_model,
    "ensemble_model": get_ensemble_model,
}


def get_available_models():
    """返回可用模型名称列表"""
    return list(MODEL_REGISTRY.keys())


def get_net(model_name, num_classes, pretrained=True):
    """选择并返回模型

    可以在此选择哪个模型用于训练

    Args:
        model_name: 模型名称
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重，默认为True
    """
    if model_name in MODEL_REGISTRY:
        print(f"Using model: {model_name}")
        return MODEL_REGISTRY[model_name](num_classes, pretrained=pretrained)
    else:
        print(f"Model {model_name} not found, using default convnextv2_base_384")
        return get_convnextv2_base_384(num_classes, pretrained=pretrained)
