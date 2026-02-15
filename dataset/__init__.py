from .data_prep import DataPreparation, setup_data


def __getattr__(name):
    """按需导入 dataloader，避免在仅使用数据准备功能时触发重依赖。"""
    if name in {"PlantDiseaseDataset", "collate_fn"}:
        from .dataloader import PlantDiseaseDataset, collate_fn

        exports = {
            "PlantDiseaseDataset": PlantDiseaseDataset,
            "collate_fn": collate_fn,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'PlantDiseaseDataset',
    'collate_fn',
    'DataPreparation',
    'setup_data',
]
