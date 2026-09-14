from __future__ import annotations

import html
from datetime import UTC, datetime

from meerax.report.builder import ReportBuilder, ReportSection

from src.services.forecast_service import StoreForecastResult
from src.services.narrative_service import NarrativeResult

GOVERNANCE_BANNER = (
    "GOVERNANCE NOTICE: This report supports a human forecast reviewer's decisions; it does not "
    "autonomously act on any forecast. Per-store narratives are LLM-generated summaries of the "
    "numeric forecast and decomposition results above them, not independently verified claims."
)


def _run_info_html(provider_name: str, store_count: int, timestamp: str) -> str:
    return (
        f"<p class='meta'>Run Info &mdash; provider: {html.escape(provider_name)} | "
        f"stores: {store_count} | generated: {html.escape(timestamp)}</p>"
    )


def build_report(
    title: str,
    results: list[StoreForecastResult],
    narratives: dict[int, NarrativeResult],
    provider_name: str = "unknown",
) -> ReportBuilder:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    report = ReportBuilder(title, subtitle=GOVERNANCE_BANNER)

    for result in results:
        narrative = narratives.get(result.store_id)
        content_parts = []
        if narrative is not None:
            content_parts.append(html.escape(narrative.trend_narrative))
            if narrative.anomaly_narrative:
                content_parts.append(html.escape(narrative.anomaly_narrative))
        content = " ".join(content_parts)

        section = ReportSection(
            title=f"Store {result.store_id}",
            content=content,
            metrics=result.metrics.to_dict(),
            figures=[result.forecast_fig, result.decomposition_fig],
        )
        report.add_section(section)

    report.add_section(
        ReportSection(
            title="Run Info",
            content=_run_info_html(provider_name, len(results), timestamp),
        )
    )
    return report
