# Modern Backbone Training Guide

This guide replaces older machine-specific remote-training notes with a portable recipe for stronger backbones.

## Recommended Baselines

| Use case | Model | Why |
| --- | --- | --- |
| fast public demo | `convnext_small` | trains quickly and has a released checkpoint |
| stronger 8 GB GPU baseline | `convnextv2_base_384` | higher-capacity modern backbone |
| transformer experiment | `eva02_large` | useful for comparison when dependencies and GPU memory allow |

## Hardware Assumptions

- GPU: 8 GB or more
- PyTorch: CUDA build when using GPU
- Python: `>=3.10,<3.15`
- Dataset layout: numeric `train/<label>/...` and `val/<label>/...`

## ConvNeXt V2 8 GB Recipe

```powershell
python main.py train `
  --model convnextv2_base_384 `
  --epochs 30 `
  --batch-size 8 `
  --dataset-path "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered" `
  --seed 888 `
  --force-train `
  --no-wandb
```

If startup image validation is too slow on an already-filtered dataset:

```powershell
python main.py train `
  --model convnextv2_base_384 `
  --epochs 30 `
  --batch-size 8 `
  --dataset-path "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered" `
  --seed 888 `
  --force-train `
  --no-wandb `
  --no-prepare `
  --no-image-validation
```

Runtime secure image loading still protects batch loading when `--no-image-validation` is used.

## Smoke Test

Run this before a long job:

```powershell
python main.py train `
  --model convnext_small `
  --epochs 1 `
  --batch-size 8 `
  --dataset-path "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered" `
  --seed 888 `
  --force-train `
  --no-wandb `
  --no-prepare `
  --no-image-validation `
  --max-train-batches 20 `
  --max-val-batches 5
```

## Evaluation

```powershell
python main.py evaluate `
  --model checkpoints/best/convnextv2_base_384/0/best_model.pth.tar `
  --model-name convnextv2_base_384 `
  --data "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered\val" `
  --batch-size 32 `
  --num-workers 0 `
  --tta-views 1 `
  --output reports/eval_convnextv2.json
```

## What to Report

- model name
- dataset source manifest
- train/validation image counts
- class count
- seed
- training command
- validation top-1/top-2/loss
- confusion matrix
- hardware and training time

## Notes

- Use W&B only when the experiment should depend on an external account; otherwise keep local JSON reports.
- Do not publish checkpoints without matching model card, dataset card, and evaluation metrics.
- Stronger accuracy on controlled datasets does not guarantee field performance.
