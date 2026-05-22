# Model Card: Plants Disease Detection Baselines

## Model Summary

This model card describes the released public baseline and the recommended longer-run baseline for this repository.

## Released Baseline

- Architecture: `convnext_small`
- Framework: PyTorch + torchvision
- Input size: `384 x 384`
- Task: plant disease image classification
- Classes: `117`
- Validation images: `7,307`
- Best checkpoint: `checkpoints/best/convnext_small/0/best_model.pth.tar`
- Suggested GitHub release tag: `convnext-small-filtered-v0.1`

### Metrics

| Split | Samples | Loss | Top-1 | Top-2 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 7,307 | 0.11197 | 95.4564% | 98.0703% |

Training progression:

| Epoch | Train loss | Train acc | Val loss | Val acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4254 | 66.9668% | 0.1681 | 90.2559% |
| 2 | 0.2073 | 86.9738% | 0.1355 | 93.5815% |
| 3 | 0.1601 | 90.4915% | 0.1120 | 95.4564% |

Reproduce the evaluation:

```powershell
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

Run the local demo:

```powershell
python app.py --download
```

## Recommended Longer Baseline

- Architecture: `convnextv2_base_384`
- Framework: PyTorch + timm
- Input size: `384 x 384`
- Task: plant disease image classification
- Default classes: synchronized from the loaded dataset
- Target model repo: `SSC-STUDIO/plants-disease-detection-convnextv2`

## Longer Training Configuration

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

Recommended run artifacts:

- `reports/dataset_stats.json`
- `reports/train_run_<date>/config.json`
- `reports/train_run_<date>/metrics.json`
- `reports/train_run_<date>/confusion_matrix.csv`
- `checkpoints/best/convnextv2_base_384/0/best_model.pth.tar`

## Evaluation

```powershell
python main.py evaluate `
  --model checkpoints/best/convnextv2_base_384/0/best_model.pth.tar `
  --output reports/eval.json
```

Record top-1 accuracy, top-k accuracy, loss, dataset version, source manifest hash, and hardware details for each public model release.

## Limitations

- Performance can drop on field images if training is dominated by controlled leaf images.
- Predictions should be treated as educational or decision-support signals, not definitive diagnosis.
- The model may inherit label noise and class imbalance from source datasets.
- The released baseline was trained for only 3 epochs as a fast public baseline; it is not a final agricultural deployment model.

## Responsible Use

Use this model for education, baseline comparison, and research prototypes. For agricultural deployment, validate on local crops, local diseases, and local imaging conditions with expert review.
