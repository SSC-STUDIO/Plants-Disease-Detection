import json
import os
from typing import Dict, Any, List, Optional

from config import config
from utils.utils import get_image_extensions

IMAGE_EXTENSIONS = get_image_extensions()


def _list_images(root: str, recursive: bool = True) -> List[str]:
    matches: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    matches.append(os.path.join(dirpath, filename))
    else:
        for filename in os.listdir(root):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                matches.append(os.path.join(root, filename))
    return matches


def summarize_dataset(
    data_path: str,
    num_classes: Optional[int] = None,
    output_file: Optional[str] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """统计数据集结构与类别分布"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    num_classes = num_classes or config.num_classes
    class_dirs = [
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ]
    labeled_dirs = [d for d in class_dirs if d.isdigit()]

    result: Dict[str, Any] = {
        "data_path": data_path,
        "total_images": 0,
        "labeled": bool(labeled_dirs),
        "class_distribution": {},
        "missing_classes": [],
    }

    if labeled_dirs:
        class_counts: Dict[int, int] = {}
        for dir_name in labeled_dirs:
            class_id = int(dir_name)
            class_path = os.path.join(data_path, dir_name)
            count = len(_list_images(class_path, recursive=True))
            class_counts[class_id] = count

        result["class_distribution"] = {
            str(k): v for k, v in sorted(class_counts.items(), key=lambda item: item[0])
        }
        result["total_images"] = sum(class_counts.values())

        expected_classes = set(range(num_classes))
        missing = sorted(list(expected_classes - set(class_counts.keys())))
        result["missing_classes"] = missing

        counts = list(class_counts.values())
        if counts:
            result["min_per_class"] = min(counts)
            result["max_per_class"] = max(counts)
            result["avg_per_class"] = sum(counts) / len(counts)

        if top_n > 0:
            top_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
            result["top_classes"] = [{"class": k, "count": v} for k, v in top_classes]
    else:
        images = _list_images(data_path, recursive=True)
        result["total_images"] = len(images)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result
