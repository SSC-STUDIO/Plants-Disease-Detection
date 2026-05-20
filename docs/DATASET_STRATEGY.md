# Dataset Strategy

The project uses a provenance-first dataset strategy. The goal is to make dataset growth useful for training while keeping public redistribution, source attribution, and license review clear.

## Principles

- Keep raw datasets outside Git.
- Track every source in `data/sources.yaml`.
- Export only sources marked `redistributable: true`.
- Preserve class mappings, source URLs, citations, and license fields.
- Filter corrupt, very small, low-information, and duplicate images before publishing.
- Keep target-domain private/self-made data as an evaluation anchor until redistribution rights are clear.

## Current Open Dataset Flow

```text
source datasets
  -> build_dataset_bundle.py
  -> export_hf_dataset.py
  -> filter_dataset_quality.py
  -> export_training_layout.py
  -> main.py train/evaluate
```

## Source Roles

| Source type | Role | Public export policy |
| --- | --- | --- |
| PlantVillage-style controlled leaf images | stable baseline and class expansion | export when license allows |
| Robustness variants | controlled stress tests for blur, contrast, JPEG, brightness | export as derived data only when source license allows |
| Field images such as PlantDoc-style data | realism and domain shift discussion | publish scripts/metadata unless redistribution is confirmed |
| Private/self-made data | target-domain evaluation anchor | do not publish without consent and license review |
| Challenge archives | research-only local comparison | do not redistribute unless terms explicitly allow |

## Recommended Local Paths

Use an environment variable instead of hard-coded machine paths:

```powershell
$env:PLANT_DATA_ROOT = "$PWD\.datasets"
```

Example directories:

```text
.datasets/
  PlantVillage-HF/
  Processed/
  PlantDisease-Open-Bundle/
  HF-PlantDisease-Open/
  HF-PlantDisease-Open-Filtered/
  PlantDisease-Open-Training-Filtered/
```

## Public Baseline Dataset

The current filtered training layout used by the released checkpoint:

| Split | Images |
| --- | ---: |
| train | 65,752 |
| val | 7,307 |
| total | 73,059 |

Classes: `117`

Known limitation: one class has only one image, so it appears in train but not validation.

## Commands

See `DATASET_CARD.md` for full rebuild commands. The most important public-export steps are:

```powershell
python tools/dataset_collector/build_dataset_bundle.py `
  --source-manifest data/sources.yaml `
  --target-dir "$env:PLANT_DATA_ROOT\PlantDisease-Open-Bundle"

python tools/dataset_collector/export_hf_dataset.py `
  --bundle-dir "$env:PLANT_DATA_ROOT\PlantDisease-Open-Bundle" `
  --output-dir "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open" `
  --dataset-repo SSC-STUDIO/plant-disease-open-dataset `
  --overwrite `
  --copy-mode link

python tools/dataset_collector/filter_dataset_quality.py `
  --input-dir "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open" `
  --output-dir "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open-Filtered" `
  --metadata-csv "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open\metadata.csv" `
  --copy-mode link `
  --overwrite `
  --near-duplicate-hamming 3

python tools/dataset_collector/export_training_layout.py `
  --input-dir "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open-Filtered" `
  --output-dir "$env:PLANT_DATA_ROOT\PlantDisease-Open-Training-Filtered" `
  --metadata-csv "$env:PLANT_DATA_ROOT\HF-PlantDisease-Open-Filtered\metadata.csv" `
  --copy-mode link `
  --overwrite `
  --stratified-val-ratio 0.1 `
  --stratified-seed 888 `
  --stratified-min-val-per-class 1
```

## Quality Checklist

Before publishing a dataset version:

- `manifest.json` exists
- `metadata.csv` exists
- `labels.json` exists
- `provenance.json` exists
- `filter_report.json` exists
- license fields are filled
- duplicate and rejection reports are kept
- dataset card explains intended use and limitations
