from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VARMAX

from src.core.interfaces import ForecastProvider


class VARMAXForecastProvider(ForecastProvider):
    def __init__(self, order: tuple[int, int] = (6, 1), trend: str = "n") -> None:
        self._order = order
        self._trend = trend
        self._result = None
        self._endog_columns: list[str] = []

    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        self._endog_columns = list(endog.columns)
        model = VARMAX(endog, exog=exog, order=self._order, trend=self._trend)
        self._result = model.fit(maxiter=1000, disp=False)

    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("VARMAXForecastProvider.forecast() called before fit()")
        values = self._result.forecast(steps=steps, exog=exog)
        return pd.DataFrame(values.to_numpy(), columns=self._endog_columns)
