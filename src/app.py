from __future__ import annotations

import argparse
import logging

import pandas as pd
from forge.llm.claude import ClaudeProvider
from forge.llm.ollama import OllamaProvider
from forge.llm.openai_provider import OpenAIProvider

from src.providers.var_provider import VARForecastProvider
from src.providers.varmax_provider import VARMAXForecastProvider
from src.providers.vecm_provider import VECMForecastProvider
from src.services.forecast_service import run_forecast_pipeline
from src.services.narrative_service import generate_narrative
from src.services.report_service import build_report

logger = logging.getLogger(__name__)

PROVIDERS: dict[str, type] = {
    "var": VARForecastProvider,
    "varmax": VARMAXForecastProvider,
    "vecm": VECMForecastProvider,
}

LLM_PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}


def top_n_stores(data: pd.DataFrame, n: int) -> list[int]:
    volumes = data.groupby("Store")["Sales"].sum().sort_values(ascending=False)
    return volumes.head(n).index.tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trend-whisperer")
    parser.add_argument("--data", default="src/data/train.csv")
    parser.add_argument("--stores", default=None, help="comma-separated store IDs; default top-N by volume")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="varmax")
    parser.add_argument("--llm-provider", choices=list(LLM_PROVIDERS.keys()), default="ollama")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--output", default="reports/model_report.html")
    args = parser.parse_args(argv)

    data = pd.read_csv(args.data)
    store_ids = (
        [int(s) for s in args.stores.split(",")] if args.stores else top_n_stores(data, args.top_n)
    )

    provider_cls = PROVIDERS[args.provider]
    results = run_forecast_pipeline(data, provider_cls, store_ids, test_size=args.test_size)

    llm = LLM_PROVIDERS[args.llm_provider]()
    narratives = {r.store_id: generate_narrative(r, llm) for r in results}

    report = build_report("TrendWhisperer — Sales Forecast Report", results, narratives)
    report.save(args.output)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
