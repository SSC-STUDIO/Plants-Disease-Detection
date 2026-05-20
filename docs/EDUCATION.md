# Education Guide

This guide turns the project into a short applied computer vision lesson.

## Lesson Path

1. Inspect the dataset source manifest in `data/sources.yaml`.
2. Run dataset statistics and discuss class imbalance.
3. Train a small reproducible baseline for a few epochs.
4. Evaluate the model and inspect the confusion matrix.
5. Discuss why controlled leaf photos and field photos create a domain shift.
6. Review `DATASET_CARD.md` and `MODEL_CARD.md` as examples of responsible AI documentation.
7. Run `app.py` and discuss model confidence on student-provided leaf images.

## Classroom Commands

```powershell
python main.py stats --data ./data/train --output reports/dataset_stats.json
python main.py train --model convnextv2_base_384 --epochs 1 --batch-size 8 --dataset-path ./data --seed 888 --force-train --no-wandb --max-train-batches 2 --max-val-batches 1
python main.py evaluate --model checkpoints/best/convnextv2_base_384/0/best_model.pth.tar --output reports/eval.json
python app.py --download
```

## Mini Course Project

Suggested 2-week assignment:

1. Rebuild the filtered dataset from `DATASET_CARD.md`.
2. Train a 1-3 epoch baseline with a fixed seed.
3. Report top-1/top-2 accuracy, confusion matrix, and the 5 most confused class pairs.
4. Compare two settings such as augmentation on/off or `convnext_small` vs `convnextv2_base_384`.
5. Add a short model card section covering intended use, limitations, and data provenance.
6. Present the Web Demo and explain one correct prediction and one failure case.

Expected deliverables:

- `reports/<run>/config.json`
- `reports/<run>/metrics.json`
- `reports/<run>/eval.json`
- `reports/<run>/confusion_matrix.csv`
- a short Markdown report with dataset and model-card notes

## Discussion Questions

- Which classes have the fewest examples, and how can that affect accuracy?
- Why can a model trained on clean leaf images fail on field photos?
- Which data sources are safe to redistribute publicly, and why?
- What should be written in a model card before sharing a checkpoint?
- How should a classroom demo communicate uncertainty and avoid overclaiming diagnosis?
