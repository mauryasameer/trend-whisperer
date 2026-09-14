# GOVERNANCE.md

## Intended Use

TrendWhisperer supports a human forecast reviewer's decisions — store managers and planners
reading sales trend/seasonality summaries and anomaly hypotheses. It does not autonomously
trigger inventory, staffing, or pricing actions. Every forecast and narrative it produces is
decision support, to be reviewed by a person before any action is taken.

## Explainability Boundary

Per-store narratives are LLM-generated plain-English summaries of the numeric forecast metrics
(RMSE/MAE/SMAPE) and, when SMAPE exceeds the anomaly threshold, a hypothesis about which of a
fixed candidate set of drivers (Promo, SchoolHoliday, CompetitionDistance) most likely explains
the miss. These are LLM-generated summaries of the numeric results shown alongside them, not
independently verified causal claims — this boundary is stated in the report itself (a permanent
banner), not only here.

## Fairness

This is a per-store retail sales forecasting model; the dataset carries no protected-attribute
data (no individual-level demographic information at all — inputs are store-level sales,
promotions, holidays, and competition distance). A disparate-impact fairness analysis does not
apply to this domain the way it would to an individual-level decision system; no fairness proxy
is reported here because none of the available features would produce a meaningful one.

## LLM Controls

Both the trend-summary and anomaly-hypothesis narrative calls pass `temperature=0` to the
configured LLM provider, so a given store's narrative is reproducible rather than randomly
sampled. The only inputs reaching the prompt are the store id and computed numeric metrics
(RMSE/MAE/SMAPE, a fixed candidate-driver list) — no retrieved documents or user-supplied free
text ever reach it, so the prompt-injection surface is assessed as low.

## Audit Trail

Every generated report embeds a "Run Info" block recording the forecast provider used, the
number of stores included, and a timestamp — so any given report is traceable to what produced
it.

## Regulatory Framing

TrendWhisperer is a retail sales-forecasting tool, not a financial-services, healthcare, or HR
system — none of SR 11-7, the EU AI Act's high-risk categories, or GDPR's special-category-data
provisions apply to this domain the way they do for a compliance-evaluation or fraud/credit
model. Noted here for completeness: this project does not currently operate under a specific
regulatory framework, and that absence is itself worth stating rather than leaving unaddressed.
