# Dataset Strategy

## Goal

Build a reusable plant-disease dataset stack that is strong enough for real training, while keeping provenance clear between:

1. The project's own self-made dataset
2. Controlled public classification data
3. In-the-wild public data
4. Archived public challenge data for later conversion

## Recommended Dataset Mix

| Dataset | Role | Why it matters | Local status |
|---------|------|----------------|--------------|
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | Controlled classification pretraining / class expansion | Large, clean, easy to train on, good for baseline accuracy | Already present locally |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset) | In-the-wild detection / harder visual domain | More realistic backgrounds and disease appearance than lab-style leaves | Already present locally |
| AI Challenger PDR2018 | Archived crop disease classification source | Useful Chinese crop disease benchmark; can be extracted and converted later | Already present locally as zip archives |
| [PlantWild / PlantWild_v2](https://tqwei05.github.io/PlantWild/) | Future in-the-wild classification upgrade | More diverse field conditions and broader disease coverage | Not yet downloaded locally |
| [PlantSeg](https://zenodo.org/records/13762907) | Future segmentation/localization upgrade | Useful when this project evolves from pure classification to lesion localization | Not yet downloaded locally |

## Self-Made Dataset

The project already contains its own dataset under:

- `C:\Users\96152\My-Project\Active\Software\Plants-Disease-Detection\data\train`
- `C:\Users\96152\My-Project\Active\Software\Plants-Disease-Detection\data\val`
- `C:\Users\96152\My-Project\Active\Software\Plants-Disease-Detection\data\test`

Current logical size:

| Split | Files | Size |
|------|------:|-----:|
| train | 31,716 | 2.826 GB |
| val | 4,539 | 0.402 GB |
| test | 8,308 | 0.484 GB |

Self-made dataset subtotal: `3.712 GB`

## Built 10GB Bundle

The reusable bundle has been assembled at:

- `C:\Users\96152\My-Project\Datasets\PlantDisease-10GB-Bundle`

It uses directory junctions to avoid wasting disk space while still giving you one stable dataset entry point.

Bundle contents:

| Source | Files | Logical size |
|--------|------:|-------------:|
| project-selfmade-train | 31,716 | 2.826 GB |
| project-selfmade-val | 4,539 | 0.402 GB |
| project-selfmade-test | 8,308 | 0.484 GB |
| PlantVillage-Dataset | 182,401 | 2.346 GB |
| PlantDoc-Object-Detection-Dataset-Linux | 5,052 | 0.914 GB |
| AI-Challenger-PDR2018 | 4 | 4.033 GB |

Bundle total: `11.005 GB`

Generated bundle metadata:

- `C:\Users\96152\My-Project\Datasets\PlantDisease-10GB-Bundle\manifest.json`
- `C:\Users\96152\My-Project\Datasets\PlantDisease-10GB-Bundle\README.md`

## Processed Datasets

Two additional ready-to-train processed datasets have now been generated locally:

### AI Challenger Converted Classification

- Path: `C:\Users\96152\My-Project\Datasets\Processed\AI-Challenger-PDR2018-Classification`
- Manifest: `C:\Users\96152\My-Project\Datasets\Processed\AI-Challenger-PDR2018-Classification\manifest.json`
- Labels: `C:\Users\96152\My-Project\Datasets\Processed\AI-Challenger-PDR2018-Classification\labels.json`
- Logical size: `3.228 GB`
- Files: `36,260`
- Splits: `train` + `val`
- Classes: `61`

Notes:

- This conversion preserves the original AI Challenger numeric class IDs as `class_00` to `class_60`.
- `testA` and `testB` were intentionally left out because they are unlabeled.

### PlantDoc Crop Classification

- Path: `C:\Users\96152\My-Project\Datasets\Processed\PlantDoc-Crops-Classification`
- Manifest: `C:\Users\96152\My-Project\Datasets\Processed\PlantDoc-Crops-Classification\manifest.json`
- Labels: `C:\Users\96152\My-Project\Datasets\Processed\PlantDoc-Crops-Classification\labels.json`
- Logical size: `0.721 GB`
- Files: `7,785`
- Splits: `train` + `test`
- Classes: `29`

Notes:

- This dataset is produced by cropping PlantDoc disease boxes with light padding.
- It is useful for injecting more realistic field imagery into a classification pipeline.
- Some very small boxes were skipped deliberately to avoid low-information crops.

## Tooling Upgrades

The built-in dataset collector now supports:

- Real low-quality filtering based on file size, minimum resolution, aspect ratio, and grayscale variance
- Real duplicate removal using SHA-256 hashing
- Headless operation without requiring `PyQt6`
- Dataset manifest generation for both local import and web collection
- Tunable CLI controls for image count, sources, quality thresholds, and deduplication

## Recommended Workflow

### 1. Clean or rebuild a self-made local dataset

```powershell
python tools/dataset_collector/app.py `
  --headless `
  --source-dir C:\path\to\raw_leaf_images `
  --output-dir C:\path\to\prepared_dataset `
  --quality-filter `
  --deduplicate `
  --enable-size-filter `
  --generate-manifest
```

### 2. Rebuild the shared 10GB bundle manifest

```powershell
python tools/dataset_collector/build_dataset_bundle.py
```

### 3. Convert AI Challenger archives into a trainable classification dataset

```powershell
python tools/dataset_collector/convert_ai_challenger.py --overwrite
```

### 4. Convert PlantDoc detection labels into classification crops

```powershell
python tools/dataset_collector/extract_plantdoc_crops.py --overwrite
```

### 5. Use the self-made data first, then expand with public sources

Recommended priority:

1. Fine-tune on the project's own `data/train`, `data/val`, `data/test`
2. Expand controlled classes from PlantVillage
3. Add harder field images from `Processed/PlantDoc-Crops-Classification`
4. Add broader benchmark data from `Processed/AI-Challenger-PDR2018-Classification`
5. Add PlantWild later if you want more real-world diversity

## Practical Advice

- Do not train only on PlantVillage if your deployment images are field photos; it is too clean.
- Keep the self-made dataset as the evaluation anchor because it is closest to your target domain.
- Treat PlantDoc and PlantWild as realism injectors.
- Keep bundle manifests under version control even if the raw datasets stay outside the repo.
