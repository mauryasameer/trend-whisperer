from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR

from src.core.interfaces import ForecastProvider


class VARForecastProvider(ForecastProvider):
    def __init__(self, maxlags: int | None = None) -> None:
        self._maxlags = maxlags
        self._result = None
        self._endog_columns: list[str] = []

    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        self._endog_columns = list(endog.columns)
        model = VAR(endog, exog=exog)
        self._result = model.fit(maxlags=self._maxlags) if self._maxlags else model.fit()

    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("VARForecastProvider.forecast() called before fit()")
        lag_order = self._result.k_ar
        forecast_input = self._result.endog[-lag_order:] if lag_order > 0 else self._result.endog
        values = self._result.forecast(y=forecast_input, steps=steps, exog_future=exog.to_numpy())
        return pd.DataFrame(values, columns=self._endog_columns)
