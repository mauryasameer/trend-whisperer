# trend-whisperer — Task Tracker

Live progress tracker. Keep this in sync with actual state.

## Backlog

- [x] Core `ForecastProvider` interface
- [x] Preprocessing service
- [x] Diagnostics service
- [x] VAR / VARMAX / VECM providers
- [x] Forecast orchestration service
- [x] Narrative service (trend + anomaly, SMAPE-gated)
- [x] Report service
- [x] Data fetch script + CLI driver
- [ ] Real end-to-end run against the actual Kaggle dataset (requires Kaggle credentials, not exercised in CI)
- [ ] Real LLM narrative review against actual Ollama output (stubbed in tests)
