# Release Notes: convnext-small-filtered-v0.1

This release is the first public training checkpoint for the open plant disease classification workflow.

## Assets

The GitHub Release includes these assets:

- `best_model.pth.tar`
- `reports/train_run_filtered_convnext_small/config.json`
- `reports/train_run_filtered_convnext_small/metrics.json`
- `reports/train_run_filtered_convnext_small/eval.json`
- `reports/train_run_filtered_convnext_small/model_hash.json`
- `reports/training_filtered_labels.json`
- `reports/train_run_filtered_convnext_small/eval/eval_20260520_075616/confusion_matrix.csv`

## Metrics

| Model | Classes | Validation images | Top-1 | Top-2 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `convnext_small` | 117 | 7,307 | 95.4564% | 98.0703% | 0.11197 |

## Reproduce

```powershell
python main.py train `
  --model convnext_small `
  --epochs 3 `
  --batch-size 8 `
  --dataset-path .datasets/PlantDisease-Open-Training-Filtered `
  --seed 888 `
  --force-train `
  --no-wandb `
  --no-prepare `
  --no-image-validation `
  --disable-augmentation `
  --no-mixup `
  --no-random-erasing `
  --disable-weighted-sampler `
  --no-ema `
  --no-gradient-checkpointing

python main.py evaluate `
  --model checkpoints/best/convnext_small/0/best_model.pth.tar `
  --model-name convnext_small `
  --data .datasets/PlantDisease-Open-Training-Filtered/val `
  --batch-size 32 `
  --num-workers 0 `
  --tta-views 1 `
  --output-dir reports/train_run_filtered_convnext_small/eval `
  --output reports/train_run_filtered_convnext_small/eval.json `
  --no-report
```

## Demo

```powershell
python app.py --download
```

Open `http://127.0.0.1:7860`.

## Notes

- This checkpoint is intended for education, research baselines, and course projects.
- It should not be used as the only basis for agricultural diagnosis.
- One unsupported MPO-formatted training image was skipped by the runtime loader during training.
