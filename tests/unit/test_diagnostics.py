import numpy as np
import pandas as pd

from src.services.diagnostics import check_stationarity, cointegration_rank, granger_causality


def test_check_stationarity_detects_stationary_series():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 200))
    result = check_stationarity(series)
    assert result["is_stationary"] is True
    assert "p_value" in result


def test_check_stationarity_detects_nonstationary_series():
    rng = np.random.default_rng(42)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 200)))
    result = check_stationarity(series)
    assert result["is_stationary"] is False


def test_granger_causality_returns_p_values_per_lag():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    y = np.roll(x, 1) + rng.normal(0, 0.1, 100)
    df = pd.DataFrame({"y": y, "x": x})
    result = granger_causality(df, maxlag=1)
    assert 1 in result
    assert "p_value" in result[1]
    assert isinstance(result[1]["p_value"], float)


def test_cointegration_rank_returns_nonnegative_int():
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(0, 1, 100))
    df = pd.DataFrame({"a": base + rng.normal(0, 0.1, 100), "b": base * 2 + rng.normal(0, 0.1, 100)})
    rank = cointegration_rank(df)
    assert isinstance(rank, int)
    assert rank >= 0
