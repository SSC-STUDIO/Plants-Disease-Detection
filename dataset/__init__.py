from .data_prep import DataPreparation, setup_data

__all__ = [
    "PlantDiseaseDataset",
    "collate_fn",
    "DataPreparation",
    "setup_data",
]


def __getattr__(name):
    """按需导入模块，减少不必要的重依赖加载。"""
    if name in {"PlantDiseaseDataset", "collate_fn"}:
        from .dataloader import PlantDiseaseDataset, collate_fn

        return {
            "PlantDiseaseDataset": PlantDiseaseDataset,
            "collate_fn": collate_fn,
        }[name]

    if name in {"DataPreparation", "setup_data"}:
        return {
            "DataPreparation": DataPreparation,
            "setup_data": setup_data,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
