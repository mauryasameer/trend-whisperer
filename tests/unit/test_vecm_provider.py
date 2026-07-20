import numpy as np
import pandas as pd
import pytest

from src.providers.vecm_provider import VECMForecastProvider


def _synthetic_data(n: int = 80):
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(0, 1, n))
    endog = pd.DataFrame(
        {
            "Sales": 100 + base + rng.normal(0, 1, n),
            "Customers": 20 + base * 0.2 + rng.normal(0, 0.5, n),
        }
    )
    exog = pd.DataFrame(
        {
            "Promo": rng.integers(0, 2, n).astype(float),
        }
    )
    return endog, exog


def test_forecast_before_fit_raises():
    provider = VECMForecastProvider()
    with pytest.raises(RuntimeError):
        provider.forecast(steps=3, exog=pd.DataFrame({"Promo": [0.0, 1.0, 0.0]}))


def test_fit_then_forecast_returns_expected_shape():
    endog, exog = _synthetic_data()
    provider = VECMForecastProvider(k_ar_diff=1, coint_rank=1)
    provider.fit(endog, exog)

    future_exog = pd.DataFrame({"Promo": [0.0, 1.0, 0.0, 1.0, 0.0]})
    result = provider.forecast(steps=5, exog=future_exog)

    assert len(result) == 5
    assert list(result.columns) == ["Sales", "Customers"]
