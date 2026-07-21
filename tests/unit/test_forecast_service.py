import numpy as np
import pandas as pd

from src.core.interfaces import ForecastProvider
from src.services.forecast_service import run_forecast_pipeline, run_store_forecast


class _StubProvider(ForecastProvider):
    def fit(self, endog, exog):
        self._fitted = True

    def forecast(self, steps, exog):
        return pd.DataFrame({"Sales": [100.0] * steps, "Customers": [10.0] * steps})


class _FailingProvider(ForecastProvider):
    def fit(self, endog, exog):
        raise ValueError("synthetic fit failure")

    def forecast(self, steps, exog):
        raise RuntimeError("should never be called")


def _make_store_df(n: int = 40, store_id: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "Store": store_id,
            "Sales": 100 + np.arange(n) + rng.normal(0, 1, n),
            "Customers": 20 + np.arange(n) * 0.2 + rng.normal(0, 0.5, n),
            "Promo": rng.integers(0, 2, n).astype(float),
            "SchoolHoliday": 0.0,
            "CompetitionDistance": 500.0,
            "DayOfWeek": [i % 7 for i in range(n)],
        }
    )


def test_run_store_forecast_returns_result_on_success():
    store_df = _make_store_df()
    result = run_store_forecast(store_id=1, store_df=store_df, provider=_StubProvider(), test_size=5)
    assert result is not None
    assert result.store_id == 1
    assert result.metrics is not None
    assert result.forecast_fig is not None
    assert result.decomposition_fig is not None


def test_run_store_forecast_returns_none_on_provider_failure():
    store_df = _make_store_df()
    result = run_store_forecast(store_id=1, store_df=store_df, provider=_FailingProvider(), test_size=5)
    assert result is None


def test_run_store_forecast_logs_diagnostics(caplog):
    store_df = _make_store_df()
    with caplog.at_level("INFO", logger="src.services.forecast_service"):
        result = run_store_forecast(store_id=1, store_df=store_df, provider=_StubProvider(), test_size=5)

    assert result is not None
    diagnostics_records = [r.message for r in caplog.records if "Diagnostics for store 1" in r.message]
    assert diagnostics_records, "expected a diagnostics log line for store 1"
    log_line = diagnostics_records[0]
    assert "stationarity" in log_line
    assert "causality" in log_line
    assert "cointegration_rank" in log_line


def test_run_forecast_pipeline_isolates_failures():
    data = pd.concat([_make_store_df(store_id=1), _make_store_df(store_id=2)], ignore_index=True)

    def provider_factory():
        return _StubProvider()

    results = run_forecast_pipeline(data, provider_factory, store_ids=[1, 2], test_size=5)
    assert len(results) == 2
    assert {r.store_id for r in results} == {1, 2}
