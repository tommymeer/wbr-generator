import pandas as pd
import numpy as np
from typing import Any

METRIC_LABELS = {
    "revenue": "Revenue",
    "pipeline_value": "Pipeline",
    "new_leads": "New Leads",
    "qualified_leads": "Qual. Leads",
    "new_customers": "New Customers",
    "churned_customers": "Churned Customers",
    "expansion_revenue": "Expansion Rev.",
    "activation_rate": "Activation Rate",
    "support_volume": "Support Vol.",
    "burn_rate": "Burn Rate",
    "runway_months": "Runway",
    # Extended schema
    "churn_rate": "Churn Rate (%)",
    "arr": "ARR",
    "mrr": "MRR",
    "ndr": "Net Dollar Retention",
    "grr": "Gross Rev. Retention",
    "cac": "CAC",
    "ltv": "LTV",
    "ltv_cac_ratio": "LTV:CAC",
    "payback_months": "CAC Payback (mo)",
    "gross_margin": "Gross Margin (%)",
    "arpu": "ARPU",
    "arpa": "ARPA",
    "gmv": "GMV",
    "take_rate": "Take Rate (%)",
    "dau": "DAU",
    "mau": "MAU",
    "dau_mau_ratio": "DAU/MAU",
    "aov": "Avg Order Value",
    "conversion_rate": "Conversion Rate (%)",
    "repeat_purchase_rate": "Repeat Purchase Rate (%)",
    "burn_multiple": "Burn Multiple",
    "rule_of_40": "Rule of 40",
    "api_calls": "API Calls",
    "gpu_utilization": "GPU Utilization (%)",
    "error_rate": "Error Rate (%)",
    "latency_ms": "Latency (ms)",
    "uptime_pct": "Uptime (%)",
    "compute_hours": "Compute Hours",
}

# Metrics where "up" is bad
INVERSE_METRICS = {
    "churned_customers", "burn_rate", "support_volume",
    "churn_rate", "cac", "payback_months", "burn_multiple",
    "error_rate", "latency_ms",
}

# Comprehensive fuzzy match hints per metric
# Rules: match on substrings in lowercased column name
# More specific hints listed first to avoid false matches
FUZZY_HINTS = {
    # Revenue variants
    "mrr": ["mrr", "monthly_recurring", "monthly recurring"],
    "arr": ["arr", "annual_recurring", "annual recurring"],
    "revenue": ["revenue", "sales", "bookings", "top_line", "topline", "income"],
    "gmv": ["gmv", "gross_merch", "gross merchandise", "transaction_volume", "transaction volume"],
    "arpu": ["arpu", "revenue_per_user", "rev_per_user", "revenue per user"],
    "arpa": ["arpa", "revenue_per_account", "rev_per_account", "revenue per account"],
    "aov": ["aov", "avg_order", "average_order", "order_value"],

    # Pipeline / leads
    "pipeline_value": ["pipeline", "pipe_value", "pipe value", "open_opps", "opportunities"],
    "new_leads": ["new_lead", "new lead", "signup", "sign_up", "prospect", "inbound", "top_of_funnel", "tofu"],
    "qualified_leads": ["qualified_lead", "qual_lead", "mql", "sql", "marketing_qualified", "sales_qualified"],

    # Customers
    "new_customers": ["new_customer", "new customer", "new_client", "won", "closed_won", "acquisition", "logos_added"],
    "churned_customers": ["churned_customer", "churned customer", "lost_customer", "cancelled_customer", "logos_lost", "logo_churn"],

    # Churn — rate vs count, important distinction
    "churn_rate": [
        "churn_rate", "churn rate", "churn_pct", "churn pct", "churn_%",
        "attrition_rate", "attrition rate", "cancellation_rate",
        "logo_churn_rate", "revenue_churn_rate",
    ],

    # Retention
    "ndr": ["ndr", "nrr", "net_dollar", "net dollar", "net_revenue_retention", "net revenue retention"],
    "grr": ["grr", "gross_revenue_retention", "gross revenue retention", "gross_dollar_retention"],
    "repeat_purchase_rate": ["repeat_purchase", "repeat purchase", "repurchase", "reorder"],

    # Unit economics
    "cac": ["cac", "customer_acquisition_cost", "customer acquisition cost", "acquisition_cost", "cost_per_acquisition", "cpa"],
    "ltv": ["ltv", "lifetime_value", "lifetime value", "clv", "customer_lifetime"],
    "ltv_cac_ratio": ["ltv_cac", "ltv:cac", "ltv/cac", "ltv cac ratio"],
    "payback_months": ["payback", "cac_payback", "months_to_payback", "recovery_months"],
    "gross_margin": ["gross_margin", "gross margin", "gm_pct", "gm_%", "margin_pct"],

    # Engagement / consumer
    "dau": ["dau", "daily_active", "daily active"],
    "mau": ["mau", "monthly_active", "monthly active"],
    "dau_mau_ratio": ["dau_mau", "dau/mau", "stickiness"],
    "conversion_rate": ["conversion_rate", "conversion rate", "cvr", "conv_rate"],
    "take_rate": ["take_rate", "take rate", "rake", "attach_rate"],

    # Expansion
    "expansion_revenue": ["expansion", "upsell", "upgrade", "ndr_revenue", "expansion_arr", "expansion_mrr"],

    # Activation
    "activation_rate": ["activation", "activated", "onboard", "onboarded", "setup_complete"],

    # Support
    "support_volume": ["support", "ticket", "support_ticket", "issue", "complaint", "help_desk", "zendesk"],

    # Financial health
    "burn_rate": ["burn_rate", "burn rate", "monthly_burn", "weekly_burn", "cash_burn", "net_burn", "spend", "opex", "operating_expense", "cost", "infrastructure_cost", "infra_cost", "cogs"],
    "runway_months": ["runway", "months_left", "cash_months", "months_of_runway", "cash_runway"],
    "burn_multiple": ["burn_multiple", "burn multiple", "efficiency_ratio"],
    "rule_of_40": ["rule_of_40", "rule of 40", "r40"],

    # AI / infra / hardware
    "api_calls": ["api_call", "api call", "api_request", "api request", "requests", "calls"],
    "gpu_utilization": ["gpu_util", "gpu utilization", "gpu_usage", "utilization_pct", "util_pct"],
    "error_rate": ["error_rate", "error rate", "failure_rate", "err_rate", "api_error"],
    "latency_ms": ["latency", "p50", "p95", "p99", "response_time", "ms"],
    "uptime_pct": ["uptime", "availability", "sla", "slo"],
    "compute_hours": ["compute_hour", "compute hour", "gpu_hour", "gpu hour", "instance_hour"],
}


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """
    Suggest known metric mapping for each CSV column via fuzzy hint matching.
    Returns {csv_col: known_metric_key or None}.
    Each known metric matched at most once (first-match wins).
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
    Rename columns per confirmed mapping.
    mapping: {original_col: target_metric_key or original_col if unmapped}
    """
    rename = {orig: target for orig, target in mapping.items() if orig != target and target}
    return df.rename(columns=rename)


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute WoW changes, trends, and derived ratios.
    Handles both mapped known columns and custom columns.
    """
    df = df.copy()
    df = df.sort_values("week").reset_index(drop=True)

    current = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    known_cols = [c for c in METRIC_LABELS if c in df.columns]
    all_numeric = df.select_dtypes(include="number").columns.tolist()
    custom_cols = [c for c in all_numeric if c not in METRIC_LABELS]

    wow_changes = {}
    directions = []

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

        directions.append({"key": col, "label": METRIC_LABELS[col], "dir": dir_, "pct": pct, "current": curr_val})

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

        directions.append({"key": col, "label": col.replace("_", " ").title(), "dir": dir_, "pct": pct, "current": curr_val})

    # Derived metrics
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

    if "mrr" in df.columns and "new_customers" in df.columns:
        mrr, nc = current["mrr"], current["new_customers"]
        derived["mrr_per_new_customer"] = round(mrr / nc, 0) if nc else None

    if "expansion_revenue" in df.columns and "revenue" in df.columns:
        exp, rev = current["expansion_revenue"], current["revenue"]
        derived["expansion_as_pct_revenue"] = round(exp / rev * 100, 1) if rev else None

    if "new_customers" in df.columns and "churned_customers" in df.columns:
        derived["net_customer_change"] = int(current["new_customers"] - current["churned_customers"])

    if "dau" in df.columns and "mau" in df.columns:
        dau, mau = current["dau"], current["mau"]
        derived["dau_mau_ratio"] = round(dau / mau * 100, 1) if mau else None

    if "ltv" in df.columns and "cac" in df.columns:
        ltv, cac = current["ltv"], current["cac"]
        derived["ltv_cac_ratio"] = round(ltv / cac, 2) if cac else None

    rev_col = next((c for c in ["revenue", "mrr"] if c in df.columns), None)
    if rev_col and "burn_rate" in df.columns:
        rev, burn = current[rev_col], current["burn_rate"]
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
                        "metric": col, "label": label, "type": "statistical",
                        "direction": direction, "z_score": round(z, 2),
                        "pct_from_mean": round(pct_from_mean, 1),
                        "current": current_val, "mean": round(mean, 2), "severity": severity,
                    })

    current = df.iloc[-1]

    # Business logic — runway
    if "runway_months" in df.columns:
        runway = current["runway_months"]
        if runway <= 6:
            anomalies.append({"metric": "runway_months", "label": "Runway", "type": "threshold",
                "direction": "critical", "severity": "high", "current": runway,
                "description": f"Runway at {runway} months — within 6-month critical window. Fundraising takes 3-6 months minimum."})
        elif runway <= 9:
            anomalies.append({"metric": "runway_months", "label": "Runway", "type": "threshold",
                "direction": "warning", "severity": "medium", "current": runway,
                "description": f"Runway at {runway} months — approaching 6-month fundraising trigger."})

    # Logo churn >= new customers
    if "churned_customers" in df.columns and "new_customers" in df.columns:
        churn, new_c = current["churned_customers"], current["new_customers"]
        if churn >= new_c:
            anomalies.append({"metric": "net_customers", "label": "Net Customer Change",
                "type": "business_logic", "direction": "negative", "severity": "high",
                "current": int(new_c - churn),
                "description": f"Churn ({int(churn)}) equals or exceeds new customers ({int(new_c)}) — negative net logo growth."})

    # NDR below 100%
    if "ndr" in df.columns:
        ndr = current["ndr"]
        if ndr < 100:
            anomalies.append({"metric": "ndr", "label": "Net Dollar Retention", "type": "threshold",
                "direction": "warning", "severity": "high", "current": ndr,
                "description": f"NDR at {ndr}% — below 100% means revenue base is decaying without new sales."})

    # LTV:CAC below 3x
    if "ltv" in df.columns and "cac" in df.columns:
        ltv, cac = current["ltv"], current["cac"]
        if cac > 0:
            ratio = ltv / cac
            if ratio < 3:
                anomalies.append({"metric": "ltv_cac", "label": "LTV:CAC", "type": "threshold",
                    "direction": "warning", "severity": "high" if ratio < 1 else "medium",
                    "current": round(ratio, 2),
                    "description": f"LTV:CAC at {ratio:.1f}x — below 3x threshold. {'Below 1x means acquiring customers destroys value.' if ratio < 1 else 'Healthy range is 3-5x.'}"})

    # Pipeline coverage < 3x revenue
    if "pipeline_value" in df.columns:
        rev_col = next((c for c in ["revenue", "mrr"] if c in df.columns), None)
        if rev_col:
            rev, pipeline = current[rev_col], current["pipeline_value"]
            if rev > 0:
                ratio = pipeline / rev
                if ratio < 3:
                    anomalies.append({"metric": "pipeline_coverage", "label": "Pipeline Coverage",
                        "type": "business_logic", "direction": "low", "severity": "medium",
                        "current": round(ratio, 1),
                        "description": f"Pipeline coverage at {ratio:.1f}x revenue — below 3x healthy threshold."})

    # Activation rate cliff
    if "activation_rate" in df.columns and len(df) >= 3:
        recent = df["activation_rate"].tail(3).values
        if len(recent) >= 3 and all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            drop = ((recent[0] - recent[-1]) / recent[0]) * 100 if recent[0] > 0 else 0
            if drop > 10:
                anomalies.append({"metric": "activation_rate", "label": "Activation Rate",
                    "type": "trend", "direction": "declining", "severity": "medium",
                    "current": round(recent[-1], 1),
                    "description": f"Activation rate declining 3 consecutive weeks, down {drop:.0f}% from 3-week high."})

    # Gross margin check — costs vs revenue
    rev_col = next((c for c in ["revenue", "mrr"] if c in df.columns), None)
    if rev_col and "burn_rate" in df.columns:
        rev, burn = current[rev_col], current["burn_rate"]
        if burn >= rev:
            anomalies.append({"metric": "gross_margin", "label": "Cost vs Revenue",
                "type": "business_logic", "direction": "critical", "severity": "high",
                "current": round((burn - rev) / rev * 100, 1),
                "description": f"Costs (${burn:,.0f}) equal or exceed revenue (${rev:,.0f}) — negative or zero margin."})

    # Error rate spike (for API/infra companies)
    if "error_rate" in df.columns and len(df) >= 3:
        recent_err = df["error_rate"].tail(3).values
        if recent_err[-1] > 1.0 and recent_err[-1] > recent_err[0] * 2:
            anomalies.append({"metric": "error_rate", "label": "Error Rate",
                "type": "business_logic", "direction": "spike", "severity": "high",
                "current": round(recent_err[-1], 2),
                "description": f"Error rate at {recent_err[-1]:.2f}% — more than doubled in 3 weeks. Platform reliability risk."})

    # GPU utilization saturation
    if "gpu_utilization" in df.columns:
        util = current["gpu_utilization"]
        if util >= 90:
            anomalies.append({"metric": "gpu_utilization", "label": "GPU Utilization",
                "type": "threshold", "direction": "critical", "severity": "high",
                "current": util,
                "description": f"GPU utilization at {util}% — near saturation. Capacity constraint imminent; latency and error rates will degrade."})

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
        lines.append("ADDITIONAL METRICS (custom columns):")
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
