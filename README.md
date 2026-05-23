# Plants Disease Detection

Open plant disease image-classification toolkit built with PyTorch. The project focuses on reproducible training, clear dataset provenance, and education-friendly workflows for learning applied computer vision in agriculture.

## Try It First

| Goal | Start here |
| --- | --- |
| Run the local Web Demo | `python app.py --download` |
| Run scriptable inference | [clients/python/README.md](clients/python/README.md) |
| Download the released model | [ConvNeXt Small filtered baseline v0.1](https://github.com/SSC-STUDIO/Plants-Disease-Detection/releases/tag/convnext-small-filtered-v0.1) |
| Reproduce a small run | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| Rebuild or publish a dataset | [DATASET_CARD.md](DATASET_CARD.md) |
| Use in a course or paper | [docs/EDUCATION.md](docs/EDUCATION.md), [docs/PAPER_PROJECT.md](docs/PAPER_PROJECT.md) |

Quick demo setup:

```powershell
git clone https://github.com/SSC-STUDIO/Plants-Disease-Detection.git
cd Plants-Disease-Detection
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-demo.txt
python app.py --download
```

Then open `http://127.0.0.1:7860`. The demo downloads the released checkpoint and label mapping when they are missing.

## Current Public Baseline

This repository now includes code, dataset tooling, training logs, a local Web Demo, and release-ready model documentation.

| Model | Dataset | Classes | Validation images | Top-1 | Top-2 | Checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `convnext_small` | filtered open training layout | 117 | 7,307 | 95.4564% | 98.0703% | GitHub Release asset |

The checkpoint and run artifacts are published as GitHub Release assets for `convnext-small-filtered-v0.1`. When you reproduce the run locally, the same files are generated under `reports/train_run_filtered_convnext_small/`:

- `config.json`
- `metrics.json`
- `eval.json`
- `eval/*/confusion_matrix.csv`

The trained checkpoint is intentionally not committed to Git because it is about 299 MB. Download it from the release asset or train it locally:

```powershell
python app.py --download
```

## What This Project Provides

- Reproducible PyTorch training for plant disease classification.
- Dataset provenance tooling that separates redistributable public data from local or restricted research data.
- Training, evaluation, prediction, and dataset statistics commands from one CLI.
- Data collection and conversion tools for local images, PlantVillage-style classification data, PlantDoc-style detection data, and archived challenge datasets.
- Model and dataset cards that make results easier to audit, publish, and teach from.
- A Gradio Web Demo for quick classroom, paper, and portfolio use.

## Start Here

- Fast setup: `docs/QUICKSTART.md`
- Web Demo: `python app.py --download`
- Python inference client: `clients/python/infer.py --download --input <image-or-folder>`
- Model details: `MODEL_CARD.md`
- Dataset details: `DATASET_CARD.md`
- Stronger-backbone training: `docs/MODERN_BACKBONE_TRAINING.md`
- Course guide: `docs/EDUCATION.md`
- Paper/course template: `docs/PAPER_PROJECT.md`
- Security examples: `docs/security_examples/`
- Contributing: `CONTRIBUTING.md`
- Citation: `CITATION.cff`

## Next Baseline Plan

The released baseline is `convnext_small`. The recommended next local experiment is designed for an 8 GB GPU such as an RTX 4060 Laptop GPU:

- Model: `convnextv2_base_384`
- Image size: `384 x 384`
- Batch size: `8`
- Epochs: `30` for the first reproducible baseline, then `50` for a longer run
- Mixed precision: enabled
- Gradient checkpointing: enabled
- Weighted sampler: enabled
- Seed: `888`
- Experiment tracking: local files by default, W&B disabled

## Install

```powershell
git clone https://github.com/SSC-STUDIO/Plants-Disease-Detection.git
cd Plants-Disease-Detection
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-demo.txt
```

Check the environment and available commands:

```powershell
python main.py version
python main.py models
```

Run the Web Demo with a local or release checkpoint:

```powershell
python app.py --download
```

Then open `http://127.0.0.1:7860`.

Run scriptable local inference:

```powershell
python clients/python/infer.py `
  --input path\to\leaf.jpg `
  --download `
  --topk 5 `
  --output reports/client_prediction.json
```

For training, first prepare a local numeric dataset with this layout:

```text
<dataset-root>/
  train/<label-id>/*.jpg
  val/<label-id>/*.jpg
```

The dataset rebuild commands below use `.datasets` by default. If Hugging Face downloads are slow or blocked, use a mirror before downloading sources:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PLANT_DATA_ROOT = "$PWD\.datasets"
```

Summarize the current training split:

```powershell
python main.py stats `
  --data "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered\train" `
  --output reports/dataset_stats.json
```

Run the next reproducible baseline after the local dataset exists:

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

Run a quick GPU smoke test before a long training job:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python main.py train `
  --model convnextv2_base_384 `
  --epochs 1 `
  --batch-size 8 `
  --dataset-path "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered" `
  --seed 888 `
  --force-train `
  --no-wandb `
  --no-prepare `
  --disable-augmentation `
  --max-train-batches 2 `
  --max-val-batches 1
```

Evaluate a trained checkpoint:

```powershell
python main.py evaluate `
  --model checkpoints/best/convnextv2_base_384/0/best_model.pth.tar `
  --model-name convnextv2_base_384 `
  --data "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered\val" `
  --output reports/eval.json
```

Run prediction with test-time augmentation after a test image directory exists:

```powershell
python main.py predict `
  --input "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered\val" `
  --tta-views 4 `
  --output ./submit/prediction_full.json `
  --output-format full `
  --topk 5 `
  --save-probs
```

## Dataset Strategy

The project uses a source manifest at `data/sources.yaml`. Each source declares:

- `name`
- `url`
- `license`
- `redistributable`
- `citation`
- `local_path`
- `splits`
- `class_mapping`

Only sources with `redistributable: true` should be exported to a public dataset repository. Restricted or unclear-license sources can still be documented and used locally when their terms allow it, but should not be copied into public releases.

Download PlantVillage through a Hugging Face mirror when GitHub access is slow or blocked:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PLANT_DATA_ROOT = "$PWD\.datasets"
python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id='mohanty/PlantVillage', repo_type='dataset', local_dir=os.path.join(os.environ.get('PLANT_DATA_ROOT', '.datasets'), 'PlantVillage-HF'), resume_download=True)"
```

Convert the mirrored archive into the numeric classification layout used by the trainer:

```powershell
python tools/dataset_collector/convert_plantvillage_hf.py `
  --source-dir "$env:PLANT_DATA_ROOT\PlantVillage-HF" `
  --output-dir "$env:PLANT_DATA_ROOT\Processed\PlantVillage-Color-Classification" `
  --overwrite

python tools/dataset_collector/convert_plantvillage_hf.py `
  --config grayscale `
  --source-dir "$env:PLANT_DATA_ROOT\PlantVillage-HF" `
  --output-dir "$env:PLANT_DATA_ROOT\Processed\PlantVillage-Grayscale-Classification" `
  --overwrite

python tools/dataset_collector/convert_plantvillage_hf.py `
  --config segmented `
  --source-dir "$env:PLANT_DATA_ROOT\PlantVillage-HF" `
  --output-dir "$env:PLANT_DATA_ROOT\Processed\PlantVillage-Segmented-Classification" `
  --overwrite
```

The current PlantVillage conversions produce `162,916` images across color, grayscale, and segmented variants. A deterministic robustness variant set adds another `87,776` images from the PlantVillage color split. These sources remain tracked as `redistributable: true` with the `CC BY-SA 3.0` source license.

Optional mirror downloads for larger local-only experiments:

```powershell
python scripts/download_hf_dataset.py `
  --repo-id avinashhm/plant-disease-classification-complete `
  --output-dir "$env:PLANT_DATA_ROOT\PlantDisease-Classification-Complete-HF" `
  --endpoint https://hf-mirror.com `
  --allow-pattern README.md `
  --allow-pattern data/**

python tools/dataset_collector/convert_hf_parquet_image_dataset.py `
  --source-dir "$env:PLANT_DATA_ROOT\PlantDisease-Classification-Complete-HF" `
  --output-dir "$env:PLANT_DATA_ROOT\Processed\PlantDisease-Classification-Complete" `
  --dataset-name PlantDisease-Classification-Complete `
  --url https://huggingface.co/datasets/avinashhm/plant-disease-classification-complete `
  --overwrite

python scripts/download_hf_dataset_files.py `
  --repo-id mbsoft31/agri-foundation-v1 `
  --output-dir "$env:PLANT_DATA_ROOT\Agri-Foundation-v1-HF" `
  --endpoint https://hf-mirror.com `
  --allow-pattern README.md `
  --allow-pattern data/** `
  --target-local-gb 3.0
```

The current full local provenance bundle is `11.177 GB` with `384,278` images. Only sources marked `redistributable: true` are exported publicly; restricted or unclear-license sources stay local.

Build a local provenance-aware bundle:

```powershell
python tools/dataset_collector/build_dataset_bundle.py `
  --source-manifest data/sources.yaml `
  --target-dir "$env:PLANT_DATA_ROOT\PlantDisease-Open-Bundle" `
  --no-validate-images `
  --no-hash
```

Export only redistributable sources into a Hugging Face-ready dataset folder:

```powershell
python tools/dataset_collector/export_hf_dataset.py `
  --bundle-dir "$env:PLANT_DATA_ROOT\PlantDisease-Open-Bundle" `
  --output-dir "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open" `
  --dataset-repo SSC-STUDIO/plant-disease-open-dataset `
  --overwrite `
  --copy-mode link
```

The export writes:

- `metadata.csv`
- `labels.json`
- `provenance.json`
- `dataset_card.md`
- `README.md`
- `data/<split>/<label>/...`

Filter the public export before publishing:

```powershell
python tools/dataset_collector/filter_dataset_quality.py `
  --input-dir "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open" `
  --output-dir "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open-Filtered" `
  --metadata-csv "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open\metadata.csv" `
  --copy-mode link `
  --overwrite `
  --min-file-size 1024 `
  --min-dimension 96 `
  --min-stddev 3.0 `
  --min-entropy 1.0 `
  --near-duplicate-hamming 3
```

The filtered export writes `filter_report.json`, `rejections.csv`, `kept_audit.csv`, and a filtered `metadata.csv`. Use `--near-duplicate-hamming 0` when you want to keep deterministic augmentation variants and remove only corrupt, low-quality, or byte-identical duplicate files.

Drop `--no-validate-images --no-hash` before a public release if you want full image integrity and duplicate scanning.

To pretrain on the expanded public PlantVillage split:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python main.py train `
  --model convnextv2_base_384 `
  --epochs 30 `
  --batch-size 8 `
  --dataset-path "$env:PLANT_DATA_ROOT\Processed\PlantVillage-Color-Classification" `
  --seed 888 `
  --force-train `
  --no-wandb `
  --no-prepare
```

## Training Artifacts

For each reproducible training run, keep these files together:

- `reports/dataset_stats.json`
- `reports/train_run_<date>/config.json`
- `reports/train_run_<date>/metrics.json`
- `reports/train_run_<date>/confusion_matrix.csv`
- `checkpoints/best/<model>/0/best_model.pth.tar`
- `MODEL_CARD.md`
- `DATASET_CARD.md`

Large raw datasets, generated reports, and checkpoints are intentionally ignored by Git. Publish datasets and models through Hugging Face or another artifact host instead of committing large binaries.

## Release Checklist

For a useful public release, publish these assets to GitHub Releases or Hugging Face:

- `best_model.pth.tar`
- `reports/train_run_filtered_convnext_small/config.json`
- `reports/train_run_filtered_convnext_small/metrics.json`
- `reports/train_run_filtered_convnext_small/eval.json`
- `reports/train_run_filtered_convnext_small/eval/*/confusion_matrix.csv`

Suggested release tag for the current baseline:

```text
convnext-small-filtered-v0.1
```

## Educational Use

This repository is suitable for lessons on:

- Image classification datasets and label quality.
- Transfer learning with modern CNN/Transformer backbones.
- Train/validation/test splits and class imbalance.
- Reproducibility, model cards, and dataset cards.
- Responsible open data practices for agriculture and field imagery.

See `docs/EDUCATION.md` for a compact lesson path.

## Project Layout

```text
config.py                    Training and path configuration
main.py                      CLI entry point
dataset/                     Data loading, preparation, and statistics
libs/                        Training, evaluation, inference, validation
models/                      Model registry and architectures
tools/dataset_collector/     Dataset import, bundle, and export tools
docs/                        Guides and educational material
tests/                       Unit and smoke tests
```

## Contributing

Issues and discussions are open. Useful contributions include:

- New redistributable plant disease data sources with license/citation metadata.
- Dataset quality checks and class mapping improvements.
- Reproducible training reports on different GPUs.
- Educational notebooks or classroom exercises.
- Model card and dataset card improvements.

Before opening a pull request, run:

```powershell
python -m compileall main.py config.py dataset libs models utils tools
pytest
```

## License

Code is released under the MIT License. Dataset licenses vary by source and are tracked in `data/sources.yaml`, bundle manifests, and dataset cards.
