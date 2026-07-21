from __future__ import annotations

import logging

from forge.llm.base import LLMProvider
from forge.llm.prompt import PromptTemplate

from src.services.forecast_service import StoreForecastResult

logger = logging.getLogger(__name__)

TREND_PROMPT = PromptTemplate(
    "Store {store_id} sales forecast summary. RMSE={rmse:.2f}, MAE={mae:.2f}, "
    "SMAPE={smape:.2f}%. Write a 2-3 sentence plain-English summary of the "
    "sales trend and seasonality for a store manager."
)

ANOMALY_PROMPT = PromptTemplate(
    "Store {store_id}'s forecast SMAPE is {smape:.2f}%, above the {threshold:.0f}% "
    "anomaly threshold. Candidate drivers available: Promo, SchoolHoliday, "
    "CompetitionDistance. In 2-3 sentences, hypothesize which of these likely "
    "explains the forecast miss."
)


class NarrativeResult:
    def __init__(self, trend_narrative: str, anomaly_narrative: str | None) -> None:
        self.trend_narrative = trend_narrative
        self.anomaly_narrative = anomaly_narrative


def generate_narrative(
    result: StoreForecastResult,
    llm: LLMProvider,
    anomaly_threshold: float = 20.0,
) -> NarrativeResult:
    try:
        trend_prompt = TREND_PROMPT.render(
            store_id=result.store_id,
            rmse=result.metrics.rmse,
            mae=result.metrics.mae,
            smape=result.metrics.smape,
        )
        trend_narrative = llm.generate(trend_prompt).content
    except Exception:
        logger.exception("Narrative generation failed for store %s", result.store_id)
        return NarrativeResult(trend_narrative="narrative unavailable", anomaly_narrative=None)

    anomaly_narrative = None
    if result.metrics.smape > anomaly_threshold:
        try:
            anomaly_prompt = ANOMALY_PROMPT.render(
                store_id=result.store_id,
                smape=result.metrics.smape,
                threshold=anomaly_threshold,
            )
            anomaly_narrative = llm.generate(anomaly_prompt).content
        except Exception:
            logger.exception("Anomaly narrative failed for store %s", result.store_id)
            anomaly_narrative = "anomaly explanation unavailable"

    return NarrativeResult(trend_narrative, anomaly_narrative)
