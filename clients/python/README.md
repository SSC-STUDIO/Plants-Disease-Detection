# Python Inference Client

This client runs local inference with the released ConvNeXt Small baseline. It can download the release checkpoint and label mapping automatically.

## Install

From the repository root:

```powershell
python -m pip install -r requirements-core.txt
```

## Single Image

```powershell
python clients/python/infer.py `
  --input path\to\leaf.jpg `
  --download `
  --topk 5 `
  --output reports/client_prediction.json
```

## Directory

```powershell
python clients/python/infer.py `
  --input path\to\images `
  --download `
  --topk 5 `
  --batch-size 16 `
  --output reports/client_predictions.json
```

## Output

The JSON output contains one item per image:

```json
[
  {
    "image": "path/to/leaf.jpg",
    "top_prediction": {
      "class_id": 0,
      "label": "Apple___healthy",
      "score": 0.98
    },
    "topk": []
  }
]
```

Use `--save-probs` to include the full probability vector.
