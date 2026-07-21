import pandas as pd
from forge.llm.base import LLMResponse

from src import app


class _StubForecastProvider:
    def fit(self, endog, exog):
        self._fitted = True

    def forecast(self, steps, exog):
        return pd.DataFrame({"Sales": [100.0] * steps, "Customers": [10.0] * steps})


class _StubLLM:
    def generate(self, prompt, system=None, **kwargs):
        return LLMResponse(content="stub narrative", model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _make_synthetic_csv(tmp_path):
    rows = []
    for store in [1, 2]:
        for day in range(40):
            rows.append(
                {
                    "Store": store,
                    "Sales": 100 + day + store,
                    "Customers": 10 + day,
                    "Promo": day % 2,
                    "SchoolHoliday": 0,
                    "CompetitionDistance": 500.0,
                    "DayOfWeek": day % 7,
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_pipeline_writes_report(tmp_path, monkeypatch):
    monkeypatch.setitem(app.PROVIDERS, "varmax", _StubForecastProvider)
    monkeypatch.setitem(app.LLM_PROVIDERS, "ollama", _StubLLM)

    data_path = _make_synthetic_csv(tmp_path)
    output_path = tmp_path / "report.html"

    exit_code = app.main(
        [
            "--data",
            str(data_path),
            "--top-n",
            "2",
            "--provider",
            "varmax",
            "--llm-provider",
            "ollama",
            "--test-size",
            "5",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    html = output_path.read_text()
    assert "Store 1" in html or "Store 2" in html
