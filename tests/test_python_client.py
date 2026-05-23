import json

import numpy as np
from PIL import Image

from clients.python.infer import format_topk, iter_image_files, write_output
from config import DefaultConfigs


def test_format_topk_uses_label_names():
    probabilities = np.array([0.1, 0.7, 0.2])
    labels = {1: "Tomato___Late_blight", 2: "Apple___healthy"}

    topk = format_topk(probabilities, labels, topk=2)

    assert topk == [
        {"class_id": 1, "label": "Tomato___Late_blight", "score": 0.7},
        {"class_id": 2, "label": "Apple___healthy", "score": 0.2},
    ]


def test_iter_image_files_supports_file_and_directory(temp_dir):
    image_path = temp_dir / "leaf.jpg"
    Image.new("RGB", (16, 16), (80, 140, 60)).save(image_path)
    (temp_dir / "notes.txt").write_text("skip", encoding="utf-8")

    cfg = DefaultConfigs()

    assert list(iter_image_files(image_path, cfg)) == [image_path]
    assert list(iter_image_files(temp_dir, cfg)) == [image_path]


def test_write_output_writes_json(temp_dir):
    output_path = temp_dir / "prediction.json"
    payload = [{"image": "leaf.jpg", "topk": []}]

    write_output(str(output_path), payload)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
