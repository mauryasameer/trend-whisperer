import matplotlib.pyplot as plt
from meerax.eval.timeseries import TimeSeriesMetrics

from src.services.forecast_service import StoreForecastResult
from src.services.narrative_service import NarrativeResult
from src.services.report_service import GOVERNANCE_BANNER, build_report


def _make_result(store_id: int) -> StoreForecastResult:
    metrics = TimeSeriesMetrics(rmse=1.0, mae=1.0, mape=1.0, smape=5.0)
    fig, _ = plt.subplots()
    return StoreForecastResult(store_id=store_id, metrics=metrics, forecast_fig=fig, decomposition_fig=fig)


def test_build_report_includes_each_store_section():
    results = [_make_result(1), _make_result(2)]
    narratives = {
        1: NarrativeResult(trend_narrative="Store 1 is trending up.", anomaly_narrative=None),
        2: NarrativeResult(trend_narrative="Store 2 is stable.", anomaly_narrative="Promo effect suspected."),
    }
    report = build_report("Test Report", results, narratives)
    html = report.to_html()

    assert "Store 1" in html
    assert "Store 2" in html
    assert "trending up" in html
    assert "Promo effect suspected" in html


def test_build_report_handles_missing_narrative():
    results = [_make_result(3)]
    report = build_report("Test Report", results, narratives={})
    html = report.to_html()
    assert "Store 3" in html


def test_build_report_includes_governance_banner_and_run_info():
    results = [_make_result(1)]
    narratives = {1: NarrativeResult(trend_narrative="Store 1 is trending up.", anomaly_narrative=None)}
    report = build_report("Test Report", results, narratives, provider_name="varmax")
    html = report.to_html()

    assert GOVERNANCE_BANNER in html
    assert "Run Info" in html
    assert "varmax" in html
    assert "stores: 1" in html


def test_build_report_escapes_narrative_content():
    results = [_make_result(1)]
    narratives = {1: NarrativeResult(trend_narrative="<script>alert(1)</script>", anomaly_narrative=None)}
    report = build_report("Test Report", results, narratives)
    html = report.to_html()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
