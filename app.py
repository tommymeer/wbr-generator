import streamlit as st
import pandas as pd
import anthropic
import json
from datetime import datetime
from analysis import compute_metrics, detect_anomalies, build_metric_narrative, suggest_mapping, apply_mapping, METRIC_LABELS
from prompt import build_wbr_prompt
st.set_page_config(
    page_title="Weekly Business Review",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --ink: #0a0a0a;
    --paper: #f5f2eb;
    --rule: #d4cfc4;
    --accent: #c8401a;
    --muted: #7a7468;
    --surface: #edeae1;
}
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: var(--paper);
    color: var(--ink);
}
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 960px; margin: 0 auto; }
/* Masthead */
.masthead {
    border-top: 3px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    padding: 1.5rem 0 1rem;
    margin-bottom: 2rem;
}
.masthead-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.03em;
    line-height: 1;
    color: var(--ink);
}
.masthead-sub {
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
/* Section headers */
.section-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 2rem;
}
/* Input fields */
textarea, input[type="text"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    border: 1px solid var(--rule) !important;
    background: var(--surface) !important;
    border-radius: 0 !important;
}
textarea:focus, input:focus {
    border-color: var(--ink) !important;
    box-shadow: none !important;
}
/* Metric pills */
.metric-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
}
.metric-pill {
    font-size: 0.7rem;
    padding: 0.25rem 0.65rem;
    border: 1px solid var(--rule);
    background: var(--surface);
    color: var(--muted);
    letter-spacing: 0.05em;
}
.metric-pill.up { border-color: #2d6a4f; color: #2d6a4f; background: #f0f7f4; }
.metric-pill.down { border-color: var(--accent); color: var(--accent); background: #fdf4f2; }
.metric-pill.flag { border-color: #b5651d; color: #b5651d; background: #fef9f5; }
/* Generate button */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #fff !important;
}
/* Output sections */
.wbr-section {
    margin-bottom: 2rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--rule);
}
.wbr-section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--ink);
    margin-bottom: 0.75rem;
}
.wbr-section-number {
    color: var(--accent);
    margin-right: 0.5rem;
}
.wbr-content {
    font-size: 0.84rem;
    line-height: 1.75;
    color: #2a2724;
}
.wbr-content p { margin-bottom: 0.6rem; }
.wbr-content ul { padding-left: 1.2rem; }
.wbr-content li { margin-bottom: 0.4rem; }
.wbr-content strong { color: var(--ink); font-weight: 500; }
/* Confidence tag */
.confidence-tag {
    display: inline-block;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    margin-bottom: 1rem;
    border: 1px dashed var(--muted);
    color: var(--muted);
}
.confidence-tag.high { border-color: #2d6a4f; color: #2d6a4f; }
.confidence-tag.medium { border-color: #b5651d; color: #b5651d; }
.confidence-tag.low { border-color: var(--accent); color: var(--accent); }
/* Upload area */
.uploadedFile { border-radius: 0 !important; }
/* Horizontal rule */
hr { border: none; border-top: 1px solid var(--rule); margin: 1.5rem 0; }
/* Anomaly callout */
.anomaly-box {
    border-left: 3px solid var(--accent);
    padding: 0.75rem 1rem;
    background: #fdf4f2;
    margin-bottom: 0.75rem;
    font-size: 0.82rem;
    line-height: 1.6;
}
/* Streamlit file uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--rule) !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
}
</style>
""", unsafe_allow_html=True)
# ── Masthead ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="masthead-title">Weekly Business Review</div>
    <div class="masthead-sub">Decision Intelligence System &nbsp;·&nbsp; Leadership Briefing</div>
</div>
""", unsafe_allow_html=True)
# ── Session state ─────────────────────────────────────────────────────────────
if "wbr_output" not in st.session_state:
    st.session_state.wbr_output = None
if "metrics_summary" not in st.session_state:
    st.session_state.metrics_summary = None
if "df" not in st.session_state:
    st.session_state.df = None
# ── Week column detection ─────────────────────────────────────────────────────
# Matches any column whose name contains one of these substrings (case-insensitive).
# Covers: week, week_ending, week_end, week_start, weekly_date, report_week,
#         date, dated, date_of, day, daily, month, monthly, quarter, quarterly,
#         year, period, time, timestamp, etc.
WEEK_SUBSTRINGS = ['week', 'date', 'day', 'month', 'quarter', 'year', 'period', 'time']
def detect_week_column(columns):
    """Return the first column whose lowercased name contains a known time substring."""
    for col in columns:
        if any(sub in col.lower() for sub in WEEK_SUBSTRINGS):
            return col
    return None
# ── Layout: two columns ───────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown('<div class="section-label">01 — Metrics Data</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div style="font-size:0.7rem; color:var(--muted); margin-top:0.5rem;">'
        'Upload any weekly business metrics CSV with a date column '
        '(e.g. <strong>week</strong>, <strong>week_ending</strong>) '
        'and numeric fields. Works with any column names — you\'ll confirm the mapping before generating.'
        '</div>',
        unsafe_allow_html=True,
    )
    MAPPING_OPTIONS = ["— unmapped —"] + list(METRIC_LABELS.keys())
    MAPPING_DISPLAY = {
        "— unmapped —": "— unmapped —",
        "revenue": "Revenue",
        "pipeline_value": "Pipeline Value",
        "new_leads": "New Leads",
        "qualified_leads": "Qualified Leads",
        "new_customers": "New Customers",
        "churned_customers": "Churned Customers",
        "expansion_revenue": "Expansion Revenue",
        "activation_rate": "Activation Rate",
        "support_volume": "Support Volume",
        "burn_rate": "Burn Rate",
        "runway_months": "Runway (months)",
    }
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            week_col = detect_week_column(raw_df.columns)
            if week_col is None:
                st.error(
                    "CSV must include a date or time column — e.g. 'week', 'week_ending', "
                    "'date', 'month', 'quarter', 'period'. None detected."
                )
            elif raw_df.select_dtypes(include="number").empty:
                st.error("CSV must include at least one numeric column.")
            else:
                # Normalize the time column to 'week' so downstream logic is unchanged
                if week_col != 'week':
                    raw_df = raw_df.rename(columns={week_col: 'week'})
                numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
                # Show a dropdown for every numeric column.
                # Pre-populate: exact schema match first, then fuzzy suggestion, then unmapped.
                st.markdown(
                    '<div style="font-size:0.7rem; color:var(--muted); margin: 0.75rem 0 0.4rem;">'
                    'Confirm what each column represents:</div>',
                    unsafe_allow_html=True,
                )
                suggestions = suggest_mapping(numeric_cols)
                confirmed_mapping = {}
                for col in numeric_cols:
                    if col in METRIC_LABELS:
                        default = col  # exact schema match — pre-select it
                    elif suggestions.get(col) in MAPPING_OPTIONS:
                        default = suggestions[col]  # fuzzy suggestion
                    else:
                        default = "— unmapped —"
                    default_idx = MAPPING_OPTIONS.index(default) if default in MAPPING_OPTIONS else 0
                    chosen = st.selectbox(
                        col,
                        options=MAPPING_OPTIONS,
                        format_func=lambda x: MAPPING_DISPLAY.get(x, x),
                        index=default_idx,
                        key=f"map_{col}",
                    )
                    confirmed_mapping[col] = chosen if chosen != "— unmapped —" else col
                if st.button("Confirm mapping", key="confirm_mapping"):
                    st.session_state.confirmed_mapping = confirmed_mapping
                    st.session_state.raw_df = raw_df
                    st.rerun()
                # Once mapping confirmed, apply and compute
                if "confirmed_mapping" in st.session_state and "raw_df" in st.session_state:
                    df = apply_mapping(st.session_state.raw_df, st.session_state.confirmed_mapping)
                    st.session_state.df = df
                    metrics = compute_metrics(df)
                    st.session_state.metrics_summary = metrics
                    known_cols = metrics.get("known_cols", [])
                    custom_cols = metrics.get("custom_cols", [])
                    absent_known = [c for c in METRIC_LABELS if c not in df.columns]
                    st.session_state.present_cols = known_cols
                    st.session_state.absent_cols = absent_known
                    st.session_state.other_cols = custom_cols
                    weeks = len(df)
                    if weeks < 2:
                        st.info("⚑ 1 week of data: no trend or anomaly analysis possible.")
                    elif weeks < 4:
                        st.info(f"⚑ {weeks} weeks of data: trend analysis directional only. Anomaly detection activates at 4+ weeks.")
                    st.markdown(
                        f'<div style="margin-top:0.75rem; font-size:0.72rem; color:var(--muted);">'
                        f'{weeks} week{"s" if weeks != 1 else ""} loaded · current week: {metrics["current_week"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    pills_html = '<div class="metric-row">'
                    for m in metrics["directions"]:
                        cls = "up" if m["dir"] == "up" else ("down" if m["dir"] == "down" else "flat")
                        arrow = "↑" if m["dir"] == "up" else ("↓" if m["dir"] == "down" else "→")
                        pills_html += f'<span class="metric-pill {cls}">{arrow} {m["label"]}</span>'
                    pills_html += '</div>'
                    st.markdown(pills_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
with right:
    st.markdown('<div class="section-label">02 — Leadership Context</div>', unsafe_allow_html=True)
    wins = st.text_area(
        "Wins this week",
        placeholder="What actually moved? Closed deals, shipped features, unblocked initiatives...",
        height=90,
        key="wins",
        label_visibility="visible",
    )
    blockers = st.text_area(
        "Blockers",
        placeholder="What's stuck, slow, or at risk? Be specific.",
        height=90,
        key="blockers",
        label_visibility="visible",
    )
    priorities = st.text_area(
        "Strategic priorities this week",
        placeholder="What does leadership need to be focused on entering next week?",
        height=90,
        key="priorities",
        label_visibility="visible",
    )
    context = st.text_area(
        "Freeform context",
        placeholder="Market signals, org dynamics, external factors, anything that doesn't fit above...",
        height=90,
        key="context",
        label_visibility="visible",
    )
# ── Generate ──────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
col_btn, _ = st.columns([1, 2])
with col_btn:
    generate = st.button("Generate Review →")
if generate:
    if st.session_state.df is None:
        st.error("Upload a CSV first.")
    elif not any([wins, blockers, priorities, context]):
        st.warning("Add at least some leadership context to generate a meaningful review.")
    else:
        metrics = st.session_state.metrics_summary
        anomalies = detect_anomalies(st.session_state.df)
        metric_narrative = build_metric_narrative(st.session_state.df, metrics)
        prompt = build_wbr_prompt(
            metrics=metrics,
            anomalies=anomalies,
            metric_narrative=metric_narrative,
            wins=wins,
            blockers=blockers,
            priorities=priorities,
            context=context,
            present_cols=st.session_state.get("present_cols", []),
            absent_cols=st.session_state.get("absent_cols", []),
            other_cols=st.session_state.get("other_cols", []),
        )
        client = anthropic.Anthropic()
        output_placeholder = st.empty()
        full_response = ""
        with st.spinner(""):
            st.markdown("""
            <div style="font-size:0.72rem; color:var(--muted); letter-spacing:0.08em; text-transform:uppercase; margin: 1rem 0 0.5rem;">
            Generating review...
            </div>""", unsafe_allow_html=True)
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=2500,
                system="""You are a senior operating executive generating a Weekly Business Review for a leadership team. 
Your job is NOT to summarize data — the data speaks for itself. Your job is to surface the decisions that need to be made, the risks that aren't yet on leadership's radar, and the questions that will drive better thinking in the room.
Be specific. Be direct. Avoid hedging language that doesn't add information. When confidence is low due to limited data, say so explicitly.
Output format: Return a JSON object with exactly these keys:
{
  "executive_summary": "string — 3-4 sentences. What is the headline business condition? What is the single most important thing leadership needs to know entering this week?",
  "key_metric_changes": ["array of strings — each entry is one metric with WoW or trend context, directional signal, and what it implies operationally. Not just the number."],
  "anomalies_and_risks": ["array of strings — each is a specific anomaly or risk with its mechanism explained. Why does this matter? What does it signal? Flag confidence level when relevant."],
  "decisions_required": ["array of strings — concrete, binary or bounded decisions leadership needs to make this week. Frame as actual decision choices, not areas of concern."],
  "leadership_questions": ["array of strings — 5-7 questions that will sharpen leadership's thinking. These should be non-obvious, probe assumptions, and create the conditions for better decisions. Not 'how do we improve X?' but questions that actually change what leadership focuses on."],
  "confidence_level": "high | medium | low",
  "confidence_note": "string — why is confidence at this level? What data is missing or ambiguous?"
}
Return only valid JSON. No markdown, no preamble.""",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
            st.session_state.wbr_output = full_response
# ── Render output ─────────────────────────────────────────────────────────────
if st.session_state.wbr_output:
    try:
        data = json.loads(st.session_state.wbr_output)
    except json.JSONDecodeError:
        # Attempt to extract JSON from possible wrapper
        import re
        match = re.search(r'\{.*\}', st.session_state.wbr_output, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            st.error("Could not parse output. Raw response:")
            st.code(st.session_state.wbr_output)
            st.stop()
    st.markdown("---")
    # Header bar
    now = datetime.now()
    confidence = data.get("confidence_level", "medium")
    conf_class = "high" if confidence == "high" else ("low" if confidence == "low" else "medium")
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1.5rem;">
        <div>
            <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.4rem; letter-spacing:-0.02em;">
                WBR — {now.strftime('%B %d, %Y')}
            </div>
            <div style="font-size:0.7rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; margin-top:0.2rem;">
                Leadership Briefing Document
            </div>
        </div>
        <div class="confidence-tag {conf_class}">Signal confidence: {confidence}</div>
    </div>
    """, unsafe_allow_html=True)
    if data.get("confidence_note"):
        st.markdown(f'<div style="font-size:0.75rem; color:var(--muted); font-style:italic; margin-bottom:1.5rem; padding:0.5rem 0.75rem; border:1px dashed var(--rule);">⚑ {data["confidence_note"]}</div>', unsafe_allow_html=True)
    # Section 1: Executive Summary
    st.markdown("""
    <div class="wbr-section">
        <div class="wbr-section-title"><span class="wbr-section-number">I.</span>Executive Summary</div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="wbr-content"><p>{data.get("executive_summary", "")}</p></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Section 2: Key Metric Changes
    st.markdown("""
    <div class="wbr-section">
        <div class="wbr-section-title"><span class="wbr-section-number">II.</span>Key Metric Changes</div>
    """, unsafe_allow_html=True)
    items_html = "<ul>"
    for item in data.get("key_metric_changes", []):
        items_html += f"<li>{item}</li>"
    items_html += "</ul>"
    st.markdown(f'<div class="wbr-content">{items_html}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Section 3: Anomalies and Risks
    st.markdown("""
    <div class="wbr-section">
        <div class="wbr-section-title"><span class="wbr-section-number">III.</span>Anomalies & Risks</div>
    """, unsafe_allow_html=True)
    for item in data.get("anomalies_and_risks", []):
        st.markdown(f'<div class="anomaly-box">{item}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Section 4: Decisions Required
    st.markdown("""
    <div class="wbr-section">
        <div class="wbr-section-title"><span class="wbr-section-number">IV.</span>Decisions Required</div>
    """, unsafe_allow_html=True)
    for i, item in enumerate(data.get("decisions_required", []), 1):
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-bottom:0.75rem; align-items:flex-start;">
            <div style="font-family:'Syne',sans-serif; font-weight:700; color:var(--accent); min-width:1.5rem; font-size:0.85rem;">{i:02d}</div>
            <div class="wbr-content" style="margin:0;">{item}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Section 5: Leadership Questions
    st.markdown("""
    <div class="wbr-section" style="border-bottom:none;">
        <div class="wbr-section-title"><span class="wbr-section-number">V.</span>Questions Leadership Should Discuss This Week</div>
    """, unsafe_allow_html=True)
    for i, q in enumerate(data.get("leadership_questions", []), 1):
        st.markdown(f"""
        <div style="padding:0.75rem 0; border-bottom:1px solid var(--rule);">
            <span style="font-size:0.68rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; margin-right:0.75rem;">Q{i}</span>
            <span style="font-size:0.85rem; line-height:1.6; color:var(--ink);">{q}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Export
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    export_text = f"""WEEKLY BUSINESS REVIEW — {now.strftime('%B %d, %Y')}
Signal Confidence: {confidence}
{data.get('confidence_note', '')}
I. EXECUTIVE SUMMARY
{data.get('executive_summary', '')}
II. KEY METRIC CHANGES
{chr(10).join(['• ' + m for m in data.get('key_metric_changes', [])])}
III. ANOMALIES & RISKS
{chr(10).join(['• ' + r for r in data.get('anomalies_and_risks', [])])}
IV. DECISIONS REQUIRED
{chr(10).join([f'{i+1:02d}. ' + d for i, d in enumerate(data.get('decisions_required', []))])}
V. LEADERSHIP QUESTIONS
{chr(10).join([f'Q{i+1}: ' + q for i, q in enumerate(data.get('leadership_questions', []))])}
"""
    st.download_button(
        label="↓ Export as Text",
        data=export_text,
        file_name=f"WBR_{now.strftime('%Y-%m-%d')}.txt",
        mime="text/plain",
    )
