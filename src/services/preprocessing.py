from __future__ import annotations

import pandas as pd


def winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    from scipy.stats import mstats

    return pd.Series(mstats.winsorize(series.to_numpy(), limits=list(limits)), index=series.index)


def one_hot_day_of_week(df: pd.DataFrame, column: str = "DayOfWeek") -> pd.DataFrame:
    dummies = pd.get_dummies(df[column], prefix=column).astype(int)
    return pd.concat([df, dummies], axis=1)


def seasonal_features(series: pd.Series, period: int = 7, prefix: str = "") -> pd.DataFrame:
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    return pd.DataFrame(
        {
            f"{prefix}Seasonality": result.seasonal,
            f"{prefix}Trend": result.trend,
        }
    )
