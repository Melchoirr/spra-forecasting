# SPRA

Dataset-agnostic implementation of Shape-Phase Retrieval with Affine Alignment
(SPRA) and its optional K-medoids candidate-bank compression variant
(SPRA-KM) for long-horizon time-series forecasting.

## Files

- `spra.py`: full-bank SPRA and compressed SPRA-KM implementation.

## Dependencies

Python 3.10 or newer with NumPy, pandas, scikit-learn, and PyTorch:

```bash
python -m pip install -r requirements.txt
```

## Input

The input is any chronologically ordered CSV with a header row. Numeric columns
are treated as time-series channels; nonnumeric columns such as timestamps are
ignored. Numeric values must be finite.

By default, the first 70% of rows form the training interval and the final 20%
form the test interval. Use `--train-end`, `--test-start`, and `--test-end`
together when exact chronological boundaries are required.

## Run

Set the seasonal period (in time steps) and the number of retrieved neighbors:

```bash
PERIOD=<seasonal-period>
TOP_K=<number-of-neighbors>
```

Full-bank SPRA:

```bash
python spra.py \
  --csv /path/to/series.csv \
  --output /path/to/full-run \
  --period "$PERIOD" \
  --top-k "$TOP_K" \
  --compression 1 \
  --device cuda:0
```

SPRA-KM retaining approximately half of the candidate bank:

```bash
python spra.py \
  --csv /path/to/series.csv \
  --output /path/to/compressed-run \
  --period "$PERIOD" \
  --top-k "$TOP_K" \
  --compression 2 \
  --max-iter 100 \
  --medoid-candidates 64 \
  --device cuda:0
```

`--top-k` accepts either one value shared by all forecast horizons or one value
per horizon. `--compression 1` uses the full candidate bank; larger integers
retain approximately the corresponding fraction of real candidates. Each
output directory must be new or empty, and the final metrics are written to
`metrics.json`.
