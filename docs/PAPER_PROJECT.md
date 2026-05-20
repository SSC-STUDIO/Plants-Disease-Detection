# Paper and Course Project Template

Use this outline for a short report, class presentation, or reproducibility appendix.

## Title

Reproducible Plant Disease Classification with Open Dataset Provenance

## Research Question

How well does a transfer-learning image classifier perform on a provenance-tracked plant disease dataset, and what are the main limitations for field use?

## Dataset

- Source manifest: `data/sources.yaml`
- Dataset card: `DATASET_CARD.md`
- Filtered training layout: `73,059` images, `117` classes
- Split: `65,752` train, `7,307` validation
- Known issue: one class has only one image and no validation example

## Method

- Model: `convnext_small` for the released baseline
- Seed: `888`
- Optimizer and loss: repository defaults unless changed
- Evaluation: top-1, top-2, loss, confusion matrix

## Current Baseline Results

| Model | Epochs | Validation images | Top-1 | Top-2 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `convnext_small` | 3 | 7,307 | 95.4564% | 98.0703% | 0.11197 |

## Required Figures or Tables

- Class distribution table from `python main.py stats`
- Confusion matrix from `python main.py evaluate`
- Top confused class pairs
- Example correct predictions and failure cases from `python app.py --download`

## Limitations

- Controlled leaf imagery can overstate field performance.
- Public sources can have label noise and inconsistent class definitions.
- Dataset licensing determines whether images can be redistributed.
- Model predictions are decision-support signals, not expert diagnosis.

## Reproducibility Checklist

- Commit hash
- Dataset manifest hash or export summary
- Training command
- Evaluation command
- Hardware
- Random seed
- Metrics JSON
- Model card and dataset card

## Citation

Cite this repository with `CITATION.cff`, and also cite the upstream datasets listed in `DATASET_CARD.md`.
