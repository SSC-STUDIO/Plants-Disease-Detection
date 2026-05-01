#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集下载和集成脚本
从多个公开数据源下载植物病害图片数据

数据源:
1. PlantVillage - 54,305张, 38类
2. PlantDoc - 2,598张, 27类 (真实环境)
3. New Plant Diseases Dataset - 87,867张, 38类
4. CCMT - 多作物病虫害数据集
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import json

# 数据集信息
DATASETS = {
    "plantvillage": {
        "name": "PlantVillage Dataset",
        "images": 54305,
        "classes": 38,
        "source": "Kaggle",
        "kaggle_dataset": "emmarex/plantdisease",
        "description": "经典植物病害数据集，室内控制环境",
    },
    "new_plant_diseases": {
        "name": "New Plant Diseases Dataset",
        "images": 87867,
        "classes": 38,
        "source": "Kaggle",
        "kaggle_dataset": "vipoooool/new-plant-diseases-dataset",
        "description": "大规模增强数据集，已划分训练/验证集",
    },
    "plantdoc": {
        "name": "PlantDoc Dataset",
        "images": 2598,
        "classes": 27,
        "source": "GitHub/Kaggle",
        "github_url": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "kaggle_dataset": "abdulhasibuddin/plant-doc-dataset",
        "description": "真实野外环境图片，背景复杂",
    },
    "plant_disease_detection": {
        "name": "Plant Disease Detection Dataset",
        "images": 0,  # 动态统计
        "classes": 0,
        "source": "Kaggle",
        "kaggle_dataset": "mgmitesh/plant-disease-detection-dataset",
        "description": "多种作物病害检测数据集",
    },
    "cassava": {
        "name": "Cassava Leaf Disease Dataset",
        "images": 9436,
        "classes": 5,
        "source": "Kaggle",
        "kaggle_dataset": "nirmalsankalana/cassava-leaf-disease-classification",
        "description": "木薯叶病害分类",
    },
}


def check_kaggle_api() -> bool:
    """检查是否配置了 Kaggle API"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_kaggle_dataset(dataset_name: str, output_dir: Path) -> bool:
    """下载 Kaggle 数据集"""
    if dataset_name not in DATASETS:
        print(f"未知数据集: {dataset_name}")
        return False

    info = DATASETS[dataset_name]
    if "kaggle_dataset" not in info:
        print(f"数据集 {dataset_name} 不是 Kaggle 数据集")
        return False

    if not check_kaggle_api():
        print("错误: 未配置 Kaggle API")
        print("请先设置 kaggle.json:")
        print("  1. 访问 https://www.kaggle.com/settings")
        print("  2. 创建 API Token")
        print("  3. 将 kaggle.json 放到 ~/.kaggle/")
        return False

    kaggle_dataset = info["kaggle_dataset"]
    output_path = output_dir / dataset_name

    print(f"\n下载 {info['name']}...")
    print(f"  Kaggle: {kaggle_dataset}")
    print(f"  输出: {output_path}")

    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", kaggle_dataset,
            "-p", str(output_path),
            "--unzip"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ 下载完成")
            return True
        else:
            print(f"  ✗ 下载失败: {result.stderr}")
            return False

    except FileNotFoundError:
        print("  ✗ 未安装 kaggle CLI，请运行: pip install kaggle")
        return False


def download_from_url(url: str, output_dir: Path) -> bool:
    """从 URL 下载文件"""
    import urllib.request

    try:
        filename = url.split("/")[-1]
        output_path = output_dir / filename

        print(f"下载: {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"保存到: {output_path}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def print_dataset_info():
    """打印所有可用数据集信息"""
    print("=" * 70)
    print("可用植物病害数据集")
    print("=" * 70)

    for key, info in DATASETS.items():
        print(f"\n【{info['name']}】")
        print(f"  图片数: {info['images']:,}")
        print(f"  类别数: {info['classes']}")
        print(f"  来源: {info['source']}")
        print(f"  描述: {info['description']}")
        if "kaggle_dataset" in info:
            print(f"  Kaggle: kaggle datasets download -d {info['kaggle_dataset']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="植物病害数据集下载工具")
    parser.add_argument("--list", action="store_true", help="列出所有可用数据集")
    parser.add_argument("--download", nargs="+", help="下载指定数据集")
    parser.add_argument("--download-all", action="store_true", help="下载所有数据集")
    parser.add_argument("--output", default="data/external", help="输出目录")

    args = parser.parse_args()

    if args.list:
        print_dataset_info()
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download_all:
        args.download = list(DATASETS.keys())

    if args.download:
        print("=" * 70)
        print("开始下载数据集")
        print("=" * 70)

        success_count = 0
        for dataset_name in args.download:
            if dataset_name in DATASETS:
                if download_kaggle_dataset(dataset_name, output_dir):
                    success_count += 1
            else:
                print(f"未知数据集: {dataset_name}")

        print(f"\n完成! 成功下载 {success_count}/{len(args.download)} 个数据集")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
