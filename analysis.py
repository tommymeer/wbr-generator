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

# Fuzzy match hints: substrings that suggest a known metric
FUZZY_HINTS = {
    "revenue": ["revenue", "mrr", "arr", "sales", "bookings", "recurring"],
    "pipeline_value": ["pipeline", "pipe"],
    "new_leads": ["lead", "signup", "sign_up", "prospect"],
    "qualified_leads": ["qualified", "qual_lead", "mql", "sql"],
    "new_customers": ["new_customer", "new_client", "closed", "won", "acquisition"],
    "churned_customers": ["churn", "attrition", "cancel", "lost"],
    "expansion_revenue": ["expansion", "upsell", "upgrade", "ndr", "net_dollar"],
    "activation_rate": ["activation", "activated", "onboard"],
    "support_volume": ["support", "ticket", "issue", "complaint"],
    "burn_rate": ["burn", "spend", "cost", "expense", "opex"],
    "runway_months": ["runway", "months_left", "cash_months"],
}


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """
    Given a list of CSV column names, suggest which known metric each maps to.
    Returns dict of {csv_col: known_metric_or_None}.
    Only maps to each known metric once (first best match wins).
    """
    suggestions = {}
    already_mapped = set()

    for col in columns:
        col_lower = col.lower()
        best_match = None
        for metric, hints in FUZZY_HINTS.items():
            if metric in already_mapped:
                continue
            if any(hint in col_lower for hint in hints):
                best_match = metric
                break
        suggestions[col] = best_match
        if best_match:
            already_mapped.add(best_match)

    return suggestions


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns per user-confirmed mapping.
    mapping: {original_col: target_metric_key or original_col if unmapped}
    Returns a new DataFrame with renamed columns.
    """
    rename = {orig: target for orig, target in mapping.items() if orig != target and target}
    return df.rename(columns=rename)


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute WoW changes, trends, and derived ratios from the CSV.
    Operates on whatever columns are present after mapping.
    """
    df = df.copy()
    df = df.sort_values("week").reset_index(drop=True)

    current = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    # Known metrics present after mapping
    known_cols = [c for c in METRIC_LABELS if c in df.columns]
    # All other numeric columns (custom/unmapped)
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    custom_cols = [c for c in all_numeric if c not in METRIC_LABELS]

    wow_changes = {}
    directions = []

    # Process known mapped columns
    for col in known_cols:
        curr_val = current[col]
        if prev is not None:
            prev_val = prev[col]
            pct = (curr_val - prev_val) / abs(prev_val) * 100 if prev_val != 0 else (float("inf") if curr_val != 0 else 0.0)
        else:
            pct = None

        wow_changes[col] = {
            "current": curr_val,
            "previous": prev[col] if prev is not None else None,
            "pct_change": pct,
            "label": METRIC_LABELS[col],
        }

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

    # Process custom (unmapped) numeric columns — compute WoW, no directional assumption
    custom_wow = {}
    for col in custom_cols:
        curr_val = current[col]
        if prev is not None:
            prev_val = prev[col]
            pct = (curr_val - prev_val) / abs(prev_val) * 100 if prev_val != 0 else 0.0
        else:
            pct = None

        custom_wow[col] = {
            "current": curr_val,
            "previous": prev[col] if prev is not None else None,
            "pct_change": pct,
            "label": col.replace("_", " ").title(),
        }

        if pct is not None:
            dir_ = "flat" if abs(pct) < 1.0 else ("up" if pct > 0 else "down")
        else:
            dir_ = "flat"

        directions.append({
            "key": col,
            "label": col.replace("_", " ").title(),
            "dir": dir_,
            "pct": pct,
            "current": curr_val,
        })

    # Derived metrics (only when both inputs are mapped)
    derived = {}

    if "new_leads" in df.columns and "qualified_leads" in df.columns:
        ql, nl = current["qualified_leads"], current["new_leads"]
        derived["lead_qualification_rate"] = round(ql / nl * 100, 1) if nl else None

    if "new_customers" in df.columns and "qualified_leads" in df.columns:
        nc, ql = current["new_customers"], current["qualified_leads"]
        derived["lead_to_close_rate"] = round(nc / ql * 100, 1) if ql else None

    if "revenue" in df.columns and "new_customers" in df.columns:
        rev, nc = current["revenue"], current["new_customers"]
        derived["revenue_per_new_customer"] = round(rev / nc, 0) if nc else None

    if "expansion_revenue" in df.columns and "revenue" in df.columns:
        exp, rev = current["expansion_revenue"], current["revenue"]
        derived["expansion_as_pct_revenue"] = round(exp / rev * 100, 1) if rev else None

    if "new_customers" in df.columns and "churned_customers" in df.columns:
        derived["net_customer_change"] = int(current["new_customers"] - current["churned_customers"])

    if "revenue" in df.columns and "burn_rate" in df.columns:
        rev, burn = current["revenue"], current["burn_rate"]
        derived["revenue_vs_burn_ratio"] = round(rev / burn, 2) if burn else None

    # Trend direction over last 4 weeks
    trends = {}
    for col in known_cols + custom_cols:
        window = df[col].tail(4).values
        if len(window) >= 3:
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
        "custom_wow": custom_wow,
        "directions": directions,
        "derived": derived,
        "trends": trends,
        "known_cols": known_cols,
        "custom_cols": custom_cols,
    }


def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    """
    Statistical + business logic anomaly detection.
    Runs on all numeric columns — both mapped and custom.
    """
    df = df.copy().sort_values("week").reset_index(drop=True)
    anomalies = []

    all_numeric = df.select_dtypes(include="number").columns.tolist()
    known_cols = [c for c in METRIC_LABELS if c in df.columns]
    custom_cols = [c for c in all_numeric if c not in METRIC_LABELS]

    # Z-score for all numeric columns
    if len(df) >= 4:
        for col in known_cols + custom_cols:
            series = df[col]
            history = series.iloc[:-1]
            current_val = series.iloc[-1]
            mean = history.mean()
            std = history.std()

            if std > 0:
                z = (current_val - mean) / std
                if abs(z) >= 2.0:
                    label = METRIC_LABELS.get(col, col.replace("_", " ").title())
                    direction = "spike" if z > 0 else "drop"
                    severity = "high" if abs(z) >= 2.5 else "medium"
                    pct_from_mean = (current_val - mean) / abs(mean) * 100 if mean != 0 else 0

                    anomalies.append({
                        "metric": col,
                        "label": label,
                        "type": "statistical",
                        "direction": direction,
                        "z_score": round(z, 2),
                        "pct_from_mean": round(pct_from_mean, 1),
                        "current": current_val,
                        "mean": round(mean, 2),
                        "severity": severity,
                    })

    current = df.iloc[-1]

    # Business logic — only on mapped columns
    if "runway_months" in df.columns:
        runway = current["runway_months"]
        if runway <= 6:
            anomalies.append({"metric": "runway_months", "label": "Runway", "type": "threshold",
                "direction": "critical", "severity": "high", "current": runway,
                "description": f"Runway at {runway} months — within 6-month critical window."})
        elif runway <= 9:
            anomalies.append({"metric": "runway_months", "label": "Runway", "type": "threshold",
                "direction": "warning", "severity": "medium", "current": runway,
                "description": f"Runway at {runway} months — approaching 6-month fundraising trigger."})

    if "churned_customers" in df.columns and "new_customers" in df.columns:
        churn, new_c = current["churned_customers"], current["new_customers"]
        if churn >= new_c:
            anomalies.append({"metric": "net_customers", "label": "Net Customer Change",
                "type": "business_logic", "direction": "negative", "severity": "high",
                "current": int(new_c - churn),
                "description": f"Churn ({int(churn)}) equals or exceeds new customers ({int(new_c)}) — negative net customer growth."})

    if "pipeline_value" in df.columns and "revenue" in df.columns:
        rev, pipeline = current["revenue"], current["pipeline_value"]
        if rev > 0:
            ratio = pipeline / rev
            if ratio < 3:
                anomalies.append({"metric": "pipeline_coverage", "label": "Pipeline Coverage",
                    "type": "business_logic", "direction": "low", "severity": "medium",
                    "current": round(ratio, 1),
                    "description": f"Pipeline coverage at {ratio:.1f}x revenue — below 3x healthy threshold."})

    if "activation_rate" in df.columns and len(df) >= 3:
        recent = df["activation_rate"].tail(3).values
        if len(recent) >= 3 and all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            drop = ((recent[0] - recent[-1]) / recent[0]) * 100 if recent[0] > 0 else 0
            if drop > 10:
                anomalies.append({"metric": "activation_rate", "label": "Activation Rate",
                    "type": "trend", "direction": "declining", "severity": "medium",
                    "current": round(recent[-1], 1),
                    "description": f"Activation rate declining 3 consecutive weeks, down {drop:.0f}% from 3-week high."})

    # Gross margin check — if both revenue and a cost column are mapped
    if "revenue" in df.columns and "burn_rate" in df.columns:
        rev, burn = current["revenue"], current["burn_rate"]
        if burn >= rev:
            anomalies.append({"metric": "gross_margin", "label": "Cost vs Revenue",
                "type": "business_logic", "direction": "critical", "severity": "high",
                "current": round((burn - rev) / rev * 100, 1),
                "description": f"Costs (${burn:,.0f}) equal or exceed revenue (${rev:,.0f}) — negative or zero margin this week."})

    return anomalies


def build_metric_narrative(df: pd.DataFrame, metrics: dict) -> str:
    """
    Build structured metric narrative for the prompt.
    Covers both mapped known columns and custom columns.
    """
    lines = []
    wow = metrics["wow_changes"]
    custom_wow = metrics.get("custom_wow", {})
    derived = metrics["derived"]
    trends = metrics["trends"]

    lines.append(f"Data covers {metrics['weeks_of_data']} weeks. Current week: {metrics['current_week']}.")
    lines.append("")

    if wow:
        lines.append("MAPPED METRICS (week-over-week):")
        for col, d in wow.items():
            curr = d["current"]
            pct = d["pct_change"]
            label = d["label"]
            trend = trends.get(col, "unknown")
            direction = f"{pct:+.1f}% WoW" if pct is not None else "first data point"
            lines.append(f"  {label}: {curr} ({direction}) | 4-week trend: {trend}")

    if custom_wow:
        lines.append("")
        lines.append("ADDITIONAL METRICS (custom columns, analyze as available):")
        for col, d in custom_wow.items():
            curr = d["current"]
            pct = d["pct_change"]
            label = d["label"]
            trend = trends.get(col, "unknown")
            direction = f"{pct:+.1f}% WoW" if pct is not None else "first data point"
            lines.append(f"  {label}: {curr} ({direction}) | 4-week trend: {trend}")

    if derived:
        lines.append("")
        lines.append("DERIVED METRICS:")
        for key, val in derived.items():
            label = key.replace("_", " ").title()
            if val is not None:
                lines.append(f"  {label}: {val}")

    return "\n".join(lines)
