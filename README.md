# merge-split-llm
A novel way to train a base model.

## Project layout
```
merge-split-llm/
├── checkpoints/        # model checkpoints (ignored by git)
├── configs/            # experiment configs
├── data/
│   ├── processed/      # processed datasets (ignored by git)
│   └── raw/            # raw datasets (ignored by git)
├── logs/               # training logs (ignored by git)
├── notebooks/          # exploration notebooks
├── scripts/            # helper scripts/entrypoints
├── src/
│   └── merge_split_llm/ # python package
└── environment.yml     # conda environment
```

## Getting started
1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate merge-split-llm
   ```
2. Install the package in editable mode once code is added:
   ```bash
   pip install -e .
   ```

## Next steps
- Add dataset preparation utilities in `scripts/`.
- Define experiment configs in `configs/`.
- Implement training code in `src/merge_split_llm/`.
