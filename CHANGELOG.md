# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.0] - 2026-07-21

### Added
- `src/core/interfaces.py` — `ForecastProvider` abstract interface
- `src/providers/` — swappable `VARForecastProvider`, `VARMAXForecastProvider` (default), `VECMForecastProvider` wrapping statsmodels
- `src/services/preprocessing.py` — winsorizing, day-of-week one-hot encoding, seasonal decomposition feature extraction
- `src/services/diagnostics.py` — stationarity, Granger causality, and Johansen cointegration checks, logged per store during orchestration
- `src/services/forecast_service.py` — per-store forecast orchestration with failure isolation
- `src/services/narrative_service.py` — LLM-generated trend narrative for every store, plus an anomaly-driver explanation gated on SMAPE > 20% (configurable)
- `src/services/report_service.py` — self-contained HTML report assembly via `forge.report.ReportBuilder`
- `scripts/fetch_data.py` — Kaggle "Rossmann Store Sales" dataset fetch, fails loudly without credentials
- `src/app.py` — CLI driver (`--provider`, `--llm-provider`, `--stores`/`--top-n`, `--test-size`, `--output`)
- 29 tests, zero real network/LLM calls

### Fixed
- Generated CI test job was missing a `pytest` install, causing every run to fail with `pytest: command not found`

[0.1.0]: https://github.com/mauryasameer/trend-whisperer/releases/tag/v0.1.0
