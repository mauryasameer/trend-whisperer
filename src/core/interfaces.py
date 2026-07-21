from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastProvider(ABC):
    """Abstract interface for all time-series forecasting backends.

    Concrete implementations live in src/providers/. Swap providers by
    changing one constructor argument — no service code changes.
    """

    @abstractmethod
    def fit(self, endog: pd.DataFrame, exog: pd.DataFrame) -> None:
        """Fit the model on historical endogenous/exogenous data."""
        ...

    @abstractmethod
    def forecast(self, steps: int, exog: pd.DataFrame) -> pd.DataFrame:
        """Forecast `steps` periods ahead given future exogenous values."""
        ...
