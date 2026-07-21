import numpy as np
import pandas as pd

from src.services.preprocessing import one_hot_day_of_week, seasonal_features, winsorize_series


def test_winsorize_series_caps_extremes():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 1000.0])
    result = winsorize_series(series, limits=(0.2, 0.2))
    assert result.max() < 1000.0
    assert len(result) == len(series)


def test_winsorize_series_preserves_index():
    series = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    result = winsorize_series(series)
    assert list(result.index) == [10, 11, 12]


def test_one_hot_day_of_week_adds_columns():
    df = pd.DataFrame({"DayOfWeek": [0, 1, 2, 0]})
    result = one_hot_day_of_week(df)
    assert "DayOfWeek_0" in result.columns
    assert "DayOfWeek_1" in result.columns
    assert "DayOfWeek_2" in result.columns
    assert result["DayOfWeek_0"].tolist() == [1, 0, 0, 1]


def test_one_hot_day_of_week_preserves_original_columns():
    df = pd.DataFrame({"DayOfWeek": [0, 1], "Sales": [100, 200]})
    result = one_hot_day_of_week(df)
    assert "Sales" in result.columns
    assert result["Sales"].tolist() == [100, 200]


def test_seasonal_features_returns_seasonality_and_trend_columns():
    values = [10 + (i % 7) + i * 0.1 for i in range(21)]
    series = pd.Series(values)
    result = seasonal_features(series, period=7, prefix="Sales")
    assert "SalesSeasonality" in result.columns
    assert "SalesTrend" in result.columns
    assert len(result) == len(series)
    assert not np.isnan(result["SalesSeasonality"]).all()
