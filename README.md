# TrendWhisperer

![Version](https://img.shields.io/badge/version-0.1.0-blue)

Per-store sales forecasting (VAR/VARMAX/VECM) over the Kaggle Rossmann Store Sales dataset,
built on [sameer-forge](https://github.com/mauryasameer/the-forge). Adds a GenAI narrative and
anomaly-explanation layer via `forge.llm`, and renders a self-contained HTML report via
`forge.report`.

## Setup

```bash
pip install -r requirements.txt
```

Set up Kaggle API credentials (`~/.kaggle/kaggle.json`) per
[Kaggle's API docs](https://www.kaggle.com/docs/api#authentication), then:

```bash
python scripts/fetch_data.py
```

## Usage

```bash
python -m src.app --provider varmax --llm-provider ollama --top-n 5
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--data` | `src/data/train.csv` | Path to the Rossmann training CSV |
| `--stores` | none (uses `--top-n`) | Comma-separated store IDs to forecast |
| `--top-n` | `5` | Number of top-by-volume stores to forecast if `--stores` is unset |
| `--provider` | `varmax` | Forecasting backend: `var` / `varmax` / `vecm` |
| `--llm-provider` | `ollama` | Narrative backend: `ollama` / `claude` / `openai` |
| `--test-size` | `100` | Holdout periods per store for evaluation |
| `--output` | `reports/model_report.html` | Report output path |

Output is a single HTML report with one section per forecasted store: metrics
(RMSE/MAE/MAPE/SMAPE), forecast and decomposition plots, an LLM-generated trend narrative, and
(when that store's SMAPE exceeds 20%) an anomaly-driver explanation.

## Architecture

- `src/core/interfaces.py` — `ForecastProvider` abstract interface
- `src/providers/` — `VARForecastProvider`, `VARMAXForecastProvider`, `VECMForecastProvider`
- `src/services/preprocessing.py` — winsorizing, day-of-week encoding, seasonal feature extraction
- `src/services/diagnostics.py` — stationarity, Granger causality, cointegration rank
- `src/services/forecast_service.py` — per-store orchestration, failure-isolated
- `src/services/narrative_service.py` — LLM trend/anomaly narrative generation
- `src/services/report_service.py` — HTML report assembly
- `scripts/fetch_data.py` — Kaggle dataset fetch

## Testing

```bash
pytest tests/ -v
```

29 tests, zero real network/LLM calls.
