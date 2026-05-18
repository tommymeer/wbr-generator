from typing import Any


def build_wbr_prompt(
    metrics: dict[str, Any],
    anomalies: list[dict],
    metric_narrative: str,
    wins: str,
    blockers: str,
    priorities: str,
    context: str,
) -> str:
    """
    Build the full user-turn prompt for WBR generation.
    Structured to give the model maximum signal with minimum noise.
    """

    # Format anomalies
    if anomalies:
        anomaly_lines = []
        for a in anomalies:
            if "description" in a:
                anomaly_lines.append(f"  [{a['severity'].upper()}] {a['label']}: {a['description']}")
            else:
                anomaly_lines.append(
                    f"  [{a['severity'].upper()}] {a['label']}: {a['direction']} — {a['pct_from_mean']:+.1f}% from historical mean (z={a['z_score']})"
                )
        anomaly_block = "\n".join(anomaly_lines)
    else:
        anomaly_block = "  No statistical anomalies detected."

    # Format leadership inputs — skip blanks
    leadership_parts = []
    if wins and wins.strip():
        leadership_parts.append(f"WINS THIS WEEK:\n{wins.strip()}")
    if blockers and blockers.strip():
        leadership_parts.append(f"BLOCKERS:\n{blockers.strip()}")
    if priorities and priorities.strip():
        leadership_parts.append(f"STRATEGIC PRIORITIES:\n{priorities.strip()}")
    if context and context.strip():
        leadership_parts.append(f"ADDITIONAL CONTEXT:\n{context.strip()}")

    leadership_block = "\n\n".join(leadership_parts) if leadership_parts else "No leadership context provided."

    weeks_note = ""
    if metrics["weeks_of_data"] < 4:
        weeks_note = f"\nNOTE: Only {metrics['weeks_of_data']} week(s) of data available. Statistical confidence is limited — flag this in confidence_level and confidence_note."
    elif metrics["weeks_of_data"] < 8:
        weeks_note = f"\nNOTE: {metrics['weeks_of_data']} weeks of data. Trend analysis is directional, not statistically robust."

    prompt = f"""Generate a Weekly Business Review for the leadership team.

━━━ METRIC DATA ━━━
{metric_narrative}{weeks_note}

━━━ DETECTED ANOMALIES ━━━
{anomaly_block}

━━━ LEADERSHIP CONTEXT ━━━
{leadership_block}

━━━ INSTRUCTIONS ━━━
Your output must prioritize:
1. DECISIONS that need to be made THIS week — concrete, bounded, actionable
2. RISKS that the data + context reveals but leadership may not be tracking yet
3. QUESTIONS that will change what leadership focuses on — not generic, not obvious

Do NOT:
- Restate data points without interpreting them
- List positives without identifying what could go wrong
- Generate generic questions like "how do we improve X?"
- Be diplomatically vague about risks

The leadership team is smart and time-constrained. Every sentence must earn its place.

Return only the JSON object as specified."""

    return prompt
