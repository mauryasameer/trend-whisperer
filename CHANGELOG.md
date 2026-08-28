# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.1.3] - 2026-08-28
### Changed
- Bumped `kaggle` requirement from `>=1.6` to `>=2.2.4` (Dependabot #15). This script only shells out to the `kaggle` CLI binary (`subprocess.run(["kaggle", "competitions", "download", "-c", ...])`), so the Python API surface never mattered — verified instead that `-c`/`-p` flags are still accepted by kaggle-cli 2.2.4 (confirmed the command reaches the real API and returns a 403 for missing credentials, not an argument-parsing error).

## [0.1.2] - 2026-08-26

### Fixed
- CI tested against a `["3.11", "3.12"]` Python version matrix, against standing instruction to
  support exactly one Python version. Single job now, Python 3.12 only. `requires-python`
  narrowed to `>=3.12`, ruff `target-version` bumped to `py312`.

## [0.1.1] - 2026-07-26

### Added
- Portfolio-ready TrendWhisperer project hero displayed in the README

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

[0.1.2]: https://github.com/mauryasameer/trend-whisperer/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/mauryasameer/trend-whisperer/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mauryasameer/trend-whisperer/releases/tag/v0.1.0
