from __future__ import annotations

from forge.report.builder import ReportBuilder, ReportSection

from src.services.forecast_service import StoreForecastResult
from src.services.narrative_service import NarrativeResult


def build_report(
    title: str,
    results: list[StoreForecastResult],
    narratives: dict[int, NarrativeResult],
) -> ReportBuilder:
    report = ReportBuilder(title)
    for result in results:
        narrative = narratives.get(result.store_id)
        content_parts = []
        if narrative is not None:
            content_parts.append(narrative.trend_narrative)
            if narrative.anomaly_narrative:
                content_parts.append(narrative.anomaly_narrative)
        content = " ".join(content_parts)

        section = ReportSection(
            title=f"Store {result.store_id}",
            content=content,
            metrics=result.metrics.to_dict(),
            figures=[result.forecast_fig, result.decomposition_fig],
        )
        report.add_section(section)
    return report
