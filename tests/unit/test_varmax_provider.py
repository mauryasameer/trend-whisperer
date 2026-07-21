import numpy as np
import pandas as pd
import pytest

from src.providers.varmax_provider import VARMAXForecastProvider


def _synthetic_data(n: int = 60):
    rng = np.random.default_rng(11)
    endog = pd.DataFrame(
        {
            "Sales": 100 + np.arange(n) * 0.5 + rng.normal(0, 2, n),
            "Customers": 20 + np.arange(n) * 0.1 + rng.normal(0, 1, n),
        }
    )
    exog = pd.DataFrame(
        {
            "Promo": rng.integers(0, 2, n).astype(float),
        }
    )
    return endog, exog


def test_forecast_before_fit_raises():
    provider = VARMAXForecastProvider()
    with pytest.raises(RuntimeError):
        provider.forecast(steps=3, exog=pd.DataFrame({"Promo": [0.0, 1.0, 0.0]}))


def test_fit_then_forecast_returns_expected_shape():
    endog, exog = _synthetic_data()
    provider = VARMAXForecastProvider(order=(1, 0), trend="n")
    provider.fit(endog, exog)

    future_exog = pd.DataFrame({"Promo": [0.0, 1.0, 0.0, 1.0, 0.0]})
    result = provider.forecast(steps=5, exog=future_exog)

    assert len(result) == 5
    assert list(result.columns) == ["Sales", "Customers"]
