import pandas as pd
import pytest

from src.core.interfaces import ForecastProvider


class _StubProvider(ForecastProvider):
    def fit(self, endog, exog):
        self._fitted = True

    def forecast(self, steps, exog):
        return pd.DataFrame({"forecast": [0.0] * steps})


def test_cannot_instantiate_abstract_provider():
    with pytest.raises(TypeError):
        ForecastProvider()


def test_stub_provider_implements_interface():
    provider = _StubProvider()
    provider.fit(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1, 2]}))
    result = provider.forecast(steps=3, exog=pd.DataFrame({"b": [1, 2, 3]}))
    assert len(result) == 3
