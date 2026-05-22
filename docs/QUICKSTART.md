# Quickstart

This guide is for a fresh clone that wants to run the released demo or reproduce a small training run.

## 1. Install

```powershell
git clone https://github.com/SSC-STUDIO/Plants-Disease-Detection.git
cd Plants-Disease-Detection
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-demo.txt
```

## 2. Run the Web Demo

This downloads the released checkpoint and label mapping if they are missing:

```powershell
python app.py --download
```

Open `http://127.0.0.1:7860`.

## 3. Evaluate the Released Model

Prepare a numeric validation directory such as:

```text
.datasets/PlantDisease-Open-Training-Filtered/val/
  0/
  1/
  ...
```

Then run:

```powershell
python main.py evaluate `
  --model checkpoints/best/convnext_small/0/best_model.pth.tar `
  --model-name convnext_small `
  --data .datasets/PlantDisease-Open-Training-Filtered/val `
  --batch-size 32 `
  --num-workers 0 `
  --tta-views 1 `
  --output-dir reports/eval_released `
  --output reports/eval_released.json
```

## 4. Rebuild the Dataset

Use `DATASET_CARD.md` for full commands. The short pattern is:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PLANT_DATA_ROOT = "$PWD\.datasets"
```

Then download sources, build the bundle, export a Hugging Face-ready folder, filter quality, and export the numeric training layout.

## 5. Run a Small Training Job

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

## 6. Report Results

For a course or paper project, include:

- dataset source manifest and license notes
- model architecture and seed
- train/validation split counts
- top-1/top-2 accuracy
- confusion matrix
- limitations and intended use
