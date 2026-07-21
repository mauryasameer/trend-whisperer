from forge.eval.timeseries import TimeSeriesMetrics
from forge.llm.base import LLMResponse

from src.services.forecast_service import StoreForecastResult
from src.services.narrative_service import generate_narrative


class _StubLLM:
    def __init__(self, content: str = "stub narrative", raise_error: bool = False):
        self._content = content
        self._raise_error = raise_error
        self.prompts_seen: list[str] = []

    def generate(self, prompt, system=None, **kwargs):
        self.prompts_seen.append(prompt)
        if self._raise_error:
            raise RuntimeError("LLM unreachable")
        return LLMResponse(content=self._content, model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _make_result(smape: float, store_id: int = 1) -> StoreForecastResult:
    metrics = TimeSeriesMetrics(rmse=1.0, mae=1.0, mape=1.0, smape=smape)
    return StoreForecastResult(store_id=store_id, metrics=metrics, forecast_fig=None, decomposition_fig=None)


def test_trend_narrative_always_generated():
    result = _make_result(smape=5.0)
    llm = _StubLLM(content="trend text")
    narrative = generate_narrative(result, llm)
    assert narrative.trend_narrative == "trend text"


def test_anomaly_narrative_generated_above_threshold():
    result = _make_result(smape=25.0)
    llm = _StubLLM(content="narrative text")
    narrative = generate_narrative(result, llm, anomaly_threshold=20.0)
    assert narrative.anomaly_narrative is not None
    assert len(llm.prompts_seen) == 2


def test_anomaly_narrative_not_generated_below_threshold():
    result = _make_result(smape=10.0)
    llm = _StubLLM(content="narrative text")
    narrative = generate_narrative(result, llm, anomaly_threshold=20.0)
    assert narrative.anomaly_narrative is None
    assert len(llm.prompts_seen) == 1


def test_trend_narrative_falls_back_on_llm_failure():
    result = _make_result(smape=5.0)
    llm = _StubLLM(raise_error=True)
    narrative = generate_narrative(result, llm)
    assert narrative.trend_narrative == "narrative unavailable"
    assert narrative.anomaly_narrative is None
