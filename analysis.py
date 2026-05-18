import pandas as pd
import numpy as np
from typing import Any

METRIC_LABELS = {
    "revenue": "Revenue",
    "pipeline_value": "Pipeline",
    "new_leads": "New Leads",
    "qualified_leads": "Qual. Leads",
    "new_customers": "New Customers",
    "churned_customers": "Churn",
    "expansion_revenue": "Expansion Rev.",
    "activation_rate": "Activation Rate",
    "support_volume": "Support Vol.",
    "burn_rate": "Burn Rate",
    "runway_months": "Runway",
}

# Metrics where "up" is bad
INVERSE_METRICS = {"churned_customers", "burn_rate", "support_volume"}

# Metrics where directional signal is less meaningful alone
CONTEXT_METRICS = {"runway_months"}


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute WoW changes, trends, and derived ratios from the CSV.
    Returns a structured summary dict for use in prompt + UI.
    """
    df = df.copy()
    df = df.sort_values("week").reset_index(drop=True)

    current = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    numeric_cols = [c for c in METRIC_LABELS if c in df.columns]

    wow_changes = {}
    directions = []

    for col in numeric_cols:
        curr_val = current[col]
        if prev is not None:
            prev_val = prev[col]
            if prev_val != 0:
                pct = (curr_val - prev_val) / abs(prev_val) * 100
            else:
                pct = float("inf") if curr_val != 0 else 0.0
        else:
            pct = None

        wow_changes[col] = {
            "current": curr_val,
            "previous": prev[col] if prev is not None else None,
            "pct_change": pct,
        }

        # Direction for UI pills
        if pct is not None:
            if abs(pct) < 1.0:
                dir_ = "flat"
            elif col in INVERSE_METRICS:
                dir_ = "down" if pct > 1 else "up"
            else:
                dir_ = "up" if pct > 0 else "down"
        else:
            dir_ = "flat"

        directions.append({
            "key": col,
            "label": METRIC_LABELS[col],
            "dir": dir_,
            "pct": pct,
            "current": curr_val,
        })

    # Derived metrics
    derived = {}

    if "new_leads" in df.columns and "qualified_leads" in df.columns:
        ql = current["qualified_leads"]
        nl = current["new_leads"]
        derived["lead_qualification_rate"] = round(ql / nl * 100, 1) if nl else None

    if "new_customers" in df.columns and "qualified_leads" in df.columns:
        nc = current["new_customers"]
        ql = current["qualified_leads"]
        derived["lead_to_close_rate"] = round(nc / ql * 100, 1) if ql else None

    if "revenue" in df.columns and "new_customers" in df.columns:
        rev = current["revenue"]
        nc = current["new_customers"]
        derived["revenue_per_new_customer"] = round(rev / nc, 0) if nc else None

    if "expansion_revenue" in df.columns and "revenue" in df.columns:
        exp = current["expansion_revenue"]
        rev = current["revenue"]
        derived["expansion_as_pct_revenue"] = round(exp / rev * 100, 1) if rev else None

    if "new_customers" in df.columns and "churned_customers" in df.columns:
        derived["net_customer_change"] = int(current["new_customers"] - current["churned_customers"])

    # Trend direction over last 4 weeks (if available)
    trends = {}
    for col in numeric_cols:
        window = df[col].tail(4).values
        if len(window) >= 3:
            # Simple linear regression slope
            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0]
            baseline = abs(window.mean()) if window.mean() != 0 else 1
            norm_slope = slope / baseline * 100
            if abs(norm_slope) < 2:
                trends[col] = "flat"
            elif col in INVERSE_METRICS:
                trends[col] = "deteriorating" if norm_slope > 0 else "improving"
            else:
                trends[col] = "improving" if norm_slope > 0 else "deteriorating"
        else:
            trends[col] = "insufficient_data"

    return {
        "current_week": str(current["week"]),
        "weeks_of_data": len(df),
        "wow_changes": wow_changes,
        "directions": directions,
        "derived": derived,
        "trends": trends,
        "raw_current": current.to_dict(),
    }


def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """
    Statistical anomaly detection using z-score and business logic checks.
    Returns list of anomaly dicts with metric, severity, and description.
    """
    df = df.copy().sort_values("week").reset_index(drop=True)
    anomalies = []

    numeric_cols = [c for c in METRIC_LABELS if c in df.columns]

    # Z-score anomalies (last week vs historical)
    if len(df) >= 4:
        for col in numeric_cols:
            series = df[col]
            history = series.iloc[:-1]
            current_val = series.iloc[-1]

            mean = history.mean()
            std = history.std()

            if std > 0:
                z = (current_val - mean) / std
                if abs(z) >= 2.0:
                    direction = "spike" if z > 0 else "drop"
                    severity = "high" if abs(z) >= 2.5 else "medium"
                    pct_from_mean = (current_val - mean) / abs(mean) * 100 if mean != 0 else 0

                    anomalies.append({
                        "metric": col,
                        "label": METRIC_LABELS[col],
                        "type": "statistical",
                        "direction": direction,
                        "z_score": round(z, 2),
                        "pct_from_mean": round(pct_from_mean, 1),
                        "current": current_val,
                        "mean": round(mean, 2),
                        "severity": severity,
                    })

    # Business logic checks
    current = df.iloc[-1]

    # Runway warning
    if "runway_months" in df.columns:
        runway = current["runway_months"]
        if runway <= 6:
            anomalies.append({
                "metric": "runway_months",
                "label": "Runway",
                "type": "threshold",
                "direction": "critical",
                "severity": "high",
                "current": runway,
                "description": f"Runway at {runway} months — within 6-month critical window.",
            })
        elif runway <= 9:
            anomalies.append({
                "metric": "runway_months",
                "label": "Runway",
                "type": "threshold",
                "direction": "warning",
                "severity": "medium",
                "current": runway,
                "description": f"Runway at {runway} months — approaching 6-month fundraising trigger.",
            })

    # Churn exceeding new customers
    if "churned_customers" in df.columns and "new_customers" in df.columns:
        churn = current["churned_customers"]
        new_c = current["new_customers"]
        if churn >= new_c:
            anomalies.append({
                "metric": "net_customers",
                "label": "Net Customer Change",
                "type": "business_logic",
                "direction": "negative",
                "severity": "high",
                "current": int(new_c - churn),
                "description": f"Churn ({int(churn)}) equals or exceeds new customers ({int(new_c)}) — negative net customer growth.",
            })

    # Pipeline-to-revenue ratio
    if "pipeline_value" in df.columns and "revenue" in df.columns:
        rev = current["revenue"]
        pipeline = current["pipeline_value"]
        if rev > 0:
            ratio = pipeline / rev
            if ratio < 3:
                anomalies.append({
                    "metric": "pipeline_coverage",
                    "label": "Pipeline Coverage",
                    "type": "business_logic",
                    "direction": "low",
                    "severity": "medium",
                    "current": round(ratio, 1),
                    "description": f"Pipeline coverage at {ratio:.1f}x revenue — below 3x healthy threshold.",
                })

    # Activation rate cliff
    if "activation_rate" in df.columns and len(df) >= 3:
        recent = df["activation_rate"].tail(3).values
        if len(recent) >= 3 and all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
            drop = ((recent[0] - recent[-1]) / recent[0]) * 100 if recent[0] > 0 else 0
            if drop > 10:
                anomalies.append({
                    "metric": "activation_rate",
                    "label": "Activation Rate",
                    "type": "trend",
                    "direction": "declining",
                    "severity": "medium",
                    "current": round(recent[-1], 1),
                    "description": f"Activation rate declining 3 consecutive weeks, down {drop:.0f}% from 3-week high.",
                })

    return anomalies


def build_metric_narrative(df: pd.DataFrame, metrics: dict) -> str:
    """
    Build a structured metric narrative string for the prompt.
    Clean, factual, no filler — just the data the model needs.
    """
    lines = []
    wow = metrics["wow_changes"]
    derived = metrics["derived"]
    trends = metrics["trends"]

    lines.append(f"Data covers {metrics['weeks_of_data']} weeks. Current week: {metrics['current_week']}.")
    lines.append("")

    lines.append("WEEK-OVER-WEEK CHANGES:")
    for col, label in METRIC_LABELS.items():
        if col not in wow:
            continue
        d = wow[col]
        curr = d["current"]
        pct = d["pct_change"]
        trend = trends.get(col, "unknown")

        if pct is not None:
            direction = f"{pct:+.1f}% WoW"
        else:
            direction = "first data point"

        lines.append(f"  {label}: {curr} ({direction}) | 4-week trend: {trend}")

    lines.append("")
    lines.append("DERIVED METRICS:")
    for key, val in derived.items():
        label = key.replace("_", " ").title()
        if val is not None:
            lines.append(f"  {label}: {val}")

    return "\n".join(lines)
