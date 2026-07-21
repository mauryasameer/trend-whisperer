from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd
from forge.eval.timeseries import TimeSeriesMetrics, evaluate_forecast
from forge.viz.timeseries import plot_decomposition, plot_forecast

from src.core.interfaces import ForecastProvider
from src.services import diagnostics, preprocessing

logger = logging.getLogger(__name__)


class StoreForecastResult:
    def __init__(self, store_id: int, metrics: TimeSeriesMetrics, forecast_fig, decomposition_fig) -> None:
        self.store_id = store_id
        self.metrics = metrics
        self.forecast_fig = forecast_fig
        self.decomposition_fig = decomposition_fig


def run_store_forecast(
    store_id: int,
    store_df: pd.DataFrame,
    provider: ForecastProvider,
    test_size: int = 100,
) -> StoreForecastResult | None:
    try:
        store_df = store_df.copy()
        store_df["Sales"] = preprocessing.winsorize_series(store_df["Sales"])
        store_df["Customers"] = preprocessing.winsorize_series(store_df["Customers"])

        sales_features = preprocessing.seasonal_features(store_df["Sales"], prefix="Sales")
        cust_features = preprocessing.seasonal_features(store_df["Customers"], prefix="Cust")
        store_df = pd.concat([store_df, sales_features, cust_features], axis=1)
        store_df = preprocessing.one_hot_day_of_week(store_df)

        train_df, test_df = store_df.iloc[:-test_size], store_df.iloc[-test_size:]
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"test_size={test_size} leaves no train or test rows for store {store_id}")

        diagnostics.check_stationarity(train_df["Sales"].dropna())

        endog_cols = ["Sales", "Customers"]
        exog_cols = [
            "Promo",
            "SchoolHoliday",
            "CompetitionDistance",
            "SalesSeasonality",
            "SalesTrend",
            "CustSeasonality",
            "CustTrend",
        ]
        train_endog = train_df[endog_cols].astype("float32").dropna()
        train_exog = train_df.loc[train_endog.index, exog_cols].astype("float32")

        provider.fit(train_endog, train_exog)

        test_exog = test_df[exog_cols].astype("float32")
        forecast_df = provider.forecast(steps=len(test_exog), exog=test_exog)

        actual = test_df["Sales"].to_numpy()[: len(forecast_df)]
        predicted = forecast_df["Sales"].to_numpy()
        metrics = evaluate_forecast(actual, predicted)

        forecast_fig = plot_forecast(actual, predicted, title=f"Store {store_id} — Sales Forecast")
        decomposition_fig = plot_decomposition(
            store_df["Sales"].dropna().to_numpy(), period=7, title=f"Store {store_id} — Decomposition"
        )

        return StoreForecastResult(store_id, metrics, forecast_fig, decomposition_fig)
    except Exception:
        logger.exception("Forecast failed for store %s", store_id)
        return None


def run_forecast_pipeline(
    data: pd.DataFrame,
    provider_factory: Callable[[], ForecastProvider],
    store_ids: list[int],
    test_size: int = 100,
) -> list[StoreForecastResult]:
    results: list[StoreForecastResult] = []
    for store_id in store_ids:
        store_df = data.loc[data["Store"] == store_id].reset_index(drop=True)
        provider = provider_factory()
        result = run_store_forecast(store_id, store_df, provider, test_size=test_size)
        if result is not None:
            results.append(result)
    return results
