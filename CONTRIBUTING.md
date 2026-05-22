# Contributing

Contributions are welcome when they improve reproducibility, dataset provenance, education value, or model quality.

## Good First Contributions

- Add a redistributable plant disease dataset source to `data/sources.yaml`.
- Improve dataset quality filters or add tests for edge cases.
- Add a short lesson, notebook, or classroom exercise under `docs/`.
- Run a baseline on a different GPU and share `config.json`, `metrics.json`, and `eval.json`.
- Improve model-card or dataset-card limitations and citation details.

## Dataset Source Requirements

Every dataset source must include:

- source name and URL
- license
- citation
- whether original images are redistributable
- local path or download instructions
- split layout
- class mapping strategy

Do not commit raw datasets, private data, unclear-license images, or generated checkpoints.

## Development Checks

Before opening a pull request:

```powershell
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-dev.txt
python -m compileall app.py main.py config.py dataset libs models utils tools tests
python -m pytest -q
```

For dataset tooling changes, also run a small temporary export and inspect the generated manifest.

## Pull Request Notes

Include:

- what changed
- how it was validated
- any dataset or license implications
- links to generated reports or release assets when relevant
