# Experiments

These scripts are “one-off” research runs. They should not be imported by `src/` code.
They load data, run an experiment, and write artifacts to `results/`.

## Common data format
The scripts expect a Bloomberg-style wide CSV like:

- Row with tickers (often row 0)
- Row with "Last Price" (often row 1)
- Row with "Dates, PX_LAST, PX_LAST, ..." (often row 2)
- Data starts on the next row

Example file: `data/merged_data.csv`

## Install
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional experiments:

```pip install -r requirements-graph.txt   # graphical lasso visualization
pip install -r requirements-ml.txt      # autoencoder experiment
```
