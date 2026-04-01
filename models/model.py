import torch
import torchvision
import torch.nn.functional as F 
from torch import nn
import os
import ssl

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
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze_matching_parameters(model, trainable_keywords):
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True


def _create_timm_model_with_retry(model_names, pretrained, **kwargs):
    last_error = None
    for model_name in model_names:
        try:
            model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
            print(f"Created model: {model_name}")
            return model
        except Exception as exc:
            last_error = exc

    if pretrained:
        print("Falling back to randomly initialized weights after pretrained download/init failure")
        for model_name in model_names:
            try:
                model = timm.create_model(model_name, pretrained=False, **kwargs)
                print(f"Created model without pretrained weights: {model_name}")
                return model
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
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    # SSL验证保持默认启用状态以确保安全
    # 如需使用特定证书，可通过环境变量 SSL_CERT_FILE 配置
    
    # 设置环境变量
    os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'pretrained')
    os.environ['HF_HOME'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'pretrained')
    
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

    try:
        model = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=pretrained,
            num_classes=0,
            features_only=False,
            out_indices=None,
        )
        print("Created EfficientNetV2-S model")
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


def get_convnextv2_base_384(num_classes, pretrained=True):
    """获取更现代的 ConvNeXt V2 Base 384 模型。"""
    if timm is None:
        print("timm is not installed, falling back to torchvision convnext_small")
        return get_convnext(num_classes, pretrained=pretrained)

    model = _create_timm_model_with_retry(
        [
            "convnextv2_base.fcmae_ft_in22k_in1k_384",
            "convnextv2_base.fcmae_ft_in22k_in1k",
        ],
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=0.2,
    )

    _freeze_all_parameters(model)
    _unfreeze_matching_parameters(model, ("stages.3", "norm", "head"))

    return model

def get_swin_transformer(num_classes, pretrained=True):
    """获取Swin Transformer模型，适用于细粒度分类任务
    Args:
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    if timm is None:
        raise ImportError("timm is required for swin_transformer model")

    model = timm.create_model('swin_small_patch4_window7_224', pretrained=pretrained)
    
    # 冻结早期层
    total_blocks = 4
    blocks_to_unfreeze = 1
    
    for name, param in model.named_parameters():
        # Using total_blocks and blocks_to_unfreeze to determine which layers to train
        if f'layers.{total_blocks - blocks_to_unfreeze}' in name or 'head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 替换分类头
    feature_dim = model.head.in_features
    model.head = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, num_classes)
    )
    
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
        pretrained (bool): 是否使用预训练权重，默认为True
    """
    if model_name in MODEL_REGISTRY:
        print(f"Using model: {model_name}")
        return MODEL_REGISTRY[model_name](num_classes, pretrained=pretrained)
    else:
        print(f"Model {model_name} not found, using default ConvNeXt V2 Base 384")
        return get_convnextv2_base_384(num_classes, pretrained=pretrained)
