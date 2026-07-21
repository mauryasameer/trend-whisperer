from __future__ import annotations

import pandas as pd

from forge.eval.timeseries import adf_stationarity


def check_stationarity(series: pd.Series) -> dict[str, float | bool]:
    return adf_stationarity(series.to_numpy())


def granger_causality(df: pd.DataFrame, maxlag: int = 1) -> dict[int, dict[str, float]]:
    from statsmodels.tsa.stattools import grangercausalitytests

    results = grangercausalitytests(df.dropna(), maxlag)
    return {lag: {"p_value": float(result[0]["ssr_ftest"][1])} for lag, result in results.items()}


def cointegration_rank(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> int:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    result = coint_johansen(df, det_order, k_ar_diff)
    traces = result.lr1
    cvts = result.cvt
    rank = 0
    for i in range(len(traces)):
        if traces[i] > cvts[i, 1]:
            rank = i + 1
    return rank
