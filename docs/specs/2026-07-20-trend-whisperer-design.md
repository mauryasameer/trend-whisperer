# TrendWhisperer

## Problem

`Projects/sale_forecasting/Time+series+forecasting.ipynb` is a monolithic notebook doing
per-store sales forecasting (VAR/VARMAX/VECM, stationarity/cointegration/causality tests,
statsmodels) against the Kaggle "Rossmann Store Sales" dataset. It reads from a hardcoded,
now-nonexistent Windows path (`D:\Rajeev\upGrad\inputDataProcessed.csv`) and can't run as-is.
It duplicates evaluation/visualization logic that already exists in `sameer-forge`
(`forge.eval.timeseries`, `forge.viz.timeseries`), and has no narrative/explanation layer over
the raw forecasts.

## Goals

- A standalone, PROJECT_STANDARDS.md-compliant, forge-dependent repo that reproduces and extends
  the notebook's forecasting pipeline: swappable VAR/VARMAX/VECM providers, stationarity/
  causality/cointegration diagnostics, forecast evaluation and plotting via `sameer-forge`.
- A GenAI narrative layer: for a configurable top-N stores, generate a plain-English trend
  summary and an anomaly-driver explanation via `forge.llm`, embedded in the final HTML report.
- Fetchable data: a script that pulls the Rossmann dataset via the Kaggle API instead of relying
  on a manually-placed file at a broken path.

## Non-goals

- Reproducing every experimental branch in the original notebook (e.g. VECM cointegration-rank
  selection stays a diagnostic step, not a swappable forecasting path).
- Building a UI — output is a single generated HTML report per run, matching forge's existing
  `ReportBuilder` pattern used elsewhere.
- Real-time/streaming forecasting — this is a batch, run-on-demand pipeline.

## Architecture

New top-level repo `/Users/sameermaurya/Downloads/dev/trend-whisperer` (own GitHub repo,
`github.com/mauryasameer/trend-whisperer`), scaffolded via `forge new trend-whisperer` so it
starts PROJECT_STANDARDS.md-compliant and `sameer-forge`-dependent from the first commit.

- **`src/core/interfaces.py`** — `ForecastProvider(ABC)`:
  - `fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None`
  - `forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame`
  Mirrors `forge.llm.base.LLMProvider`'s abstract-interface-then-swap pattern.

- **`src/providers/`** — three implementations behind `ForecastProvider`:
  - `var_provider.py` — `VARForecastProvider`, wraps `statsmodels.tsa.api.VAR`.
  - `varmax_provider.py` — `VARMAXForecastProvider`, wraps `statsmodels.tsa.api.VARMAX`
    (order `(6,1,0)`, `trend='n')` — the notebook's actual final forecasting model, and the
    CLI's default.
  - `vecm_provider.py` — `VECMForecastProvider`, wraps `statsmodels.tsa.api.VECM`.

- **`src/services/preprocessing.py`** — pure functions extracted from the notebook's inline
  feature engineering: winsorizing (`scipy.stats.mstats.winsorize`), day-of-week one-hot
  encoding, seasonal-decomposition-derived features (`SalesSeasonality`, `SalesTrend`,
  `CustSeasonality`, `CustTrend`).

- **`src/services/diagnostics.py`** — stationarity (wraps
  `forge.eval.timeseries.adf_stationarity`, replacing the notebook's hand-rolled `adfuller`
  call), Granger causality (`statsmodels.tsa.stattools.grangercausalitytests`), and Johansen
  cointegration (`coint_johansen`, `select_coint_rank`). Analysis-only — results are logged and
  inform preprocessing (e.g. whether to difference a series), not swappable providers.

- **`src/services/forecast_service.py`** — orchestrates, per selected store: preprocess →
  diagnose → `ForecastProvider.fit()`/`.forecast()` → `forge.eval.timeseries.evaluate_forecast()`
  → `forge.viz.timeseries.plot_forecast()` / `plot_decomposition()`. Catches and logs per-store
  failures (e.g. insufficient observations to fit) without aborting the run; failed stores are
  skipped and noted in the final report.

- **`src/services/narrative_service.py`** — for the top-N stores by sales volume (configurable,
  default 5), builds two `forge.llm.PromptTemplate`s and calls a `forge.llm.LLMProvider`
  (default `OllamaProvider`, swappable via `--llm-provider`):
  1. Trend/seasonality narrative from the store's forecast + decomposition.
  2. Anomaly-driver hypothesis, generated whenever that store's own `TimeSeriesMetrics.smape`
     (from `forecast_service.py`'s `evaluate_forecast` call) exceeds a configurable threshold
     (default 20%), using the `Promo`/`SchoolHoliday`/`CompetitionDistance` exog columns as
     candidate drivers. Below the threshold, only the trend narrative is generated.
  LLM failures (e.g. Ollama not running) are caught per-store, logged, and that store's report
  section falls back to metrics/plots only with a "narrative unavailable" note.

- **`src/services/report_service.py`** — assembles one `forge.report.ReportBuilder` per run,
  one `ReportSection` per store: metrics table (from `TimeSeriesMetrics.to_dict()`), forecast
  and decomposition figures, narrative text, anomaly text.

- **`scripts/fetch_data.py`** — `kaggle competitions download -c rossmann-store-sales`, unzips
  into `src/data/` (gitignored per PROJECT_STANDARDS.md). Fails loudly with setup instructions
  if `~/.kaggle/kaggle.json` is missing — no silent fallback.

- **`src/app.py`** — CLI driver. Flags: `--stores` (default: top-5 by sales volume, or an
  explicit comma-separated list of store IDs), `--provider` (`var`/`varmax`/`vecm`, default
  `varmax`), `--llm-provider` (`ollama`/`claude`/`openai`, default `ollama`). Runs
  `fetch → preprocess → diagnose → forecast → narrate → report`, writing
  `reports/model_report.html`.

## Data Flow

`scripts/fetch_data.py` → `src/data/rossmann.csv` (gitignored) → `preprocessing.py` loads,
winsorizes, and engineers features per store → `diagnostics.py` runs stationarity/causality/
cointegration checks (logged, informs differencing decisions, non-blocking) →
`forecast_service.py` selects the top-N stores by volume, fits the configured
`ForecastProvider`, forecasts, scores via `evaluate_forecast` → `narrative_service.py` turns
each selected store's forecast + metrics + exog into two LLM calls → `report_service.py` renders
one HTML report with all per-store sections.

## Error Handling

- `scripts/fetch_data.py`: missing Kaggle credentials → print setup instructions, exit 1. No
  silent fallback, no bundled/committed dataset.
- `forecast_service.py`: per-store fit/forecast failures are caught, logged via `logging`
  (never `print`), and that store is skipped in the report rather than aborting the run.
- `narrative_service.py`: per-store LLM failures are caught, logged, and produce a "narrative
  unavailable" fallback in that store's report section — forecasting output is never blocked by
  the LLM layer being unreachable.

## Testing

- `tests/unit/test_preprocessing.py` — winsorizing, one-hot encoding, seasonal feature
  extraction against small synthetic fixtures (3-5 rows).
- `tests/unit/test_diagnostics.py` — stationarity/causality/cointegration wrappers against
  known stationary vs. non-stationary synthetic series.
- `tests/unit/test_providers.py` — each `ForecastProvider`'s fit/forecast against a small
  synthetic multivariate series; asserts output shape and that `forecast()` requires `fit()`
  first.
- `tests/unit/test_forecast_service.py` — orchestration logic against a stub `ForecastProvider`;
  asserts one bad store doesn't abort the run.
- `tests/unit/test_narrative_service.py` — stub `LLMProvider` (matching forge's own test
  pattern); asserts prompt construction, the SMAPE-threshold gating (anomaly prompt fires above
  20%, trend-only below it), and graceful fallback on LLM failure.
- `tests/integration/test_pipeline.py` — end-to-end on a small synthetic dataset (no real
  Kaggle download, no real LLM call); asserts a report file is written with the expected number
  of sections.

## Versioning

New repo starts at `0.1.0` per `forge new`'s scaffold default. Standard PROJECT_STANDARDS.md
branch/PR/version rules apply from the first feature branch onward.
