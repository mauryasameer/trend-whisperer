from __future__ import annotations

import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from src.core.interfaces import ForecastProvider


class VECMForecastProvider(ForecastProvider):
    def __init__(self, k_ar_diff: int = 1, coint_rank: int = 1) -> None:
        self._k_ar_diff = k_ar_diff
        self._coint_rank = coint_rank
        self._result = None
        self._endog_columns: list[str] = []

    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        self._endog_columns = list(endog.columns)
        model = VECM(endog, exog=exog, k_ar_diff=self._k_ar_diff, coint_rank=self._coint_rank)
        self._result = model.fit()

    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        if self._result is None:
            raise RuntimeError("VECMForecastProvider.forecast() called before fit()")
        values = self._result.predict(steps=steps, exog_fc=exog.to_numpy())
        return pd.DataFrame(values, columns=self._endog_columns)
