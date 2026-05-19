# Weekly Business Review Generator

Most business review tools give leadership more information. This one gives them fewer decisions to make.

**[→ Try it live](https://wbr-generator-7uckwuf8k4jobxlkrvgxmu.streamlit.app)**

---

## The problem

Weekly business reviews fail in a predictable way. Someone exports data into a dashboard. The dashboard shows everything. Leadership spends the meeting discussing what the numbers mean instead of what to do about them.

The bottleneck isn't data — it's interpretation. Specifically: which of these signals require a decision this week, which are noise, and what questions will actually change how we think about the business?

That's a judgment problem, not a data problem. Dashboards can't solve it.

---

## What this does

Upload your weekly metrics CSV and add three sentences of leadership context. The tool generates a structured briefing focused on:

- **Decisions required this week** — concrete, bounded, with a clear choice framed
- **Anomalies and risks** — with mechanism explained, not just flagged
- **Leadership questions** — non-obvious, designed to change what the room focuses on

No charts. No summaries. No restating what the numbers already say.

The intelligence is in decision surfacing and question generation. Summarization is a commodity.

---

## Architecture

Most AI demos are: upload file → prompt LLM → output. The failure mode is hallucination and false confidence.

This tool separates two layers deliberately:

**Deterministic layer** (`analysis.py`) — Python computes what actually happened. Week-over-week changes, 4-week trend slopes, derived ratios (lead-to-close rate, pipeline coverage, net customer change), and statistical anomaly detection via z-score against historical baseline. The model never touches raw numbers — it receives structured facts.

**Reasoning layer** (`app.py` + `prompt.py`) — The model interprets implications, surfaces decisions, and generates questions. It's explicitly instructed not to summarize, not to reason about data that isn't present, and to scale confidence to data quality.

The prompt includes a DATA AVAILABILITY block that lists exactly which columns exist and which don't — preventing the model from hallucinating analysis on fields that aren't in the CSV.

---

## Anomaly detection

Two types run before every generation:

**Statistical** — z-score against 4-week historical baseline. Fires at |z| ≥ 2.0, high severity at 2.5. Applied to every numeric column.

**Business logic** — hardcoded heuristics that don't require probabilistic inference:
- Runway ≤ 9 months (warning) and ≤ 6 months (critical)
- Churn ≥ new customers (negative net growth)
- Pipeline coverage < 3x revenue
- Activation rate declining 3+ consecutive weeks

Executive workflows are partly statistical, partly judgmental. Both layers are necessary.

---

## Confidence and ambiguity handling

The tool explicitly handles thin data rather than pretending confidence it doesn't have:

- 1 week of data: no trend or anomaly analysis, brief output, low confidence flagged
- 2–3 weeks: directional only, uncertainty called out explicitly
- 4+ weeks: statistical anomaly detection activates
- 8+ weeks: full trend analysis

Missing columns are surfaced as informational notes, not errors. The model is instructed not to reason about absent fields.

---

## CSV format

Upload any weekly metrics CSV with a **week** column and numeric fields. The tool works with whatever you track — no fixed schema required. When you upload a CSV with non-standard column names, a mapping step appears that auto-suggests matches using fuzzy name matching (e.g. **revenue_mrr** → MRR, **churn_rate_pct** → Churn Rate, **cac_dollars** → CAC). You confirm or correct each mapping before generating. Columns left unmapped pass through as custom fields and are still analyzed for trends and anomalies — they just don't trigger the business logic heuristics that require knowing what a column represents.

Common fields it understands natively:

| Column | Description |
|---|---|
| `week` | Week identifier (any format) |
| `revenue` | Weekly revenue |
| `pipeline_value` | Qualified pipeline value |
| `new_leads` | Leads created |
| `qualified_leads` | Leads marked qualified |
| `new_customers` | Customers closed |
| `churned_customers` | Customers churned |
| `expansion_revenue` | Upsell/expansion revenue |
| `activation_rate` | Product activation rate (0–1) |
| `support_volume` | Support tickets opened |
| `burn_rate` | Weekly cash burn |
| `runway_months` | Cash runway remaining |

Custom columns are passed through and analyzed as available.

---

## Stack

- Python + Pandas + NumPy — deterministic metric computation and anomaly detection
- Anthropic API (Claude Sonnet) — reasoning, decision surfacing, streaming output
- Streamlit — frontend and deployment

---

## Local setup

```bash
git clone https://github.com/tommymeer/wbr-generator
cd wbr-generator
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → main file: `app.py`
3. In app Settings → Secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

---

## Extending

**New metrics** — add to `METRIC_LABELS` and `INVERSE_METRICS` in `analysis.py`. Inverse metrics are those where an increase is bad (churn, burn rate, support volume).

**New output sections** — update the JSON schema in the system prompt in `app.py` and add a render block below the existing sections.

**New anomaly checks** — add business logic to `detect_anomalies()` in `analysis.py`. Statistical detection is automatic for any numeric column.
