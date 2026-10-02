"""Funnel Friction Analyzer — Streamlit UI.

UI only: all calculations live in ``funnel.py`` and ``opportunity_model.py``.
Run with ``streamlit run app.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from funnel import (
    LEAKAGE_METRICS,
    clean_funnel,
    compute_metrics,
    largest_leakage,
    transition_table,
    validate_funnel,
)
from opportunity_model import (
    DEFAULT_EFFORT,
    EFFORT_MAX,
    EFFORT_MIN,
    build_summary,
    ranked,
    size_opportunities,
)

ROOT = Path(__file__).parent
SAMPLE_FUNNEL = ROOT / "sample_funnel.csv"

# Synthetic default effort (1 = easy, 5 = hard) for the sample funnel's transitions.
SAMPLE_EFFORT = {
    "Landing Page → Form Start": 1,
    "Form Start → Form Complete": 2,
    "Form Complete → Qualified Lead": 3,
    "Qualified Lead → Opportunity": 3,
    "Opportunity → Customer": 4,
}

BAR_COLOR = "#2a78d6"
LEAK_COLOR = "#d03b3b"

st.set_page_config(page_title="Funnel Friction Analyzer", page_icon="🔻", layout="wide")


# ------------------------------------------------------------------ helpers

def pct(x: float, digits: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x:.{digits}%}"


def funnel_chart(metrics: pd.DataFrame, leak_to_index: int) -> go.Figure:
    colors = [LEAK_COLOR if i == leak_to_index else BAR_COLOR for i in range(len(metrics))]
    labels = [f"⚠ {s}" if i == leak_to_index else s for i, s in enumerate(metrics["stage"])]
    fig = go.Figure(go.Funnel(
        y=labels,
        x=metrics["users"],
        textposition="auto",
        texttemplate="%{value:,.0f} · %{percentPrevious:.1%} of previous",
        marker={"color": colors},
        connector={"line": {"width": 0}, "fillcolor": "rgba(137,135,129,0.15)"},
        hovertemplate=("<b>%{y}</b><br>Users: %{x:,.0f}<br>"
                       "Step conversion: %{percentPrevious:.1%}<br>"
                       "Cumulative: %{percentInitial:.2%}<extra></extra>"),
    ))
    fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=420)
    return fig


def md_safe(text: str) -> str:
    """Escape characters Streamlit markdown would treat as strikethrough (~) or LaTeX ($)."""
    return text.replace("~", r"\~").replace("$", r"\$")


def metrics_display(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Stage": metrics["stage"],
        "Users": metrics["users"].map("{:,.0f}".format),
        "Step conversion": metrics["step_conversion"].map(pct),
        "Drop-off": metrics["drop_off"].map(pct),
        "Users lost": metrics["users_lost"].map("{:,.0f}".format),
        "Cumulative conversion": metrics["cumulative_conversion"].map(lambda v: pct(v, 2)),
    })


def read_csv(upload) -> pd.DataFrame | None:
    try:
        return pd.read_csv(upload)
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        st.error(f"Couldn't read that CSV: {exc}")
        return None


# ------------------------------------------------------------------ sidebar

st.sidebar.title("🔻 Funnel Friction Analyzer")
source = st.sidebar.radio("Funnel data", ["Sample funnel", "Upload CSV"], horizontal=True)

raw = pd.read_csv(SAMPLE_FUNNEL)
editor_key = "funnel-sample"
if source == "Upload CSV":
    upload = st.sidebar.file_uploader("CSV with columns stage, users (top → bottom)", type="csv")
    if upload is None:
        st.info("Upload a CSV with columns `stage, users`, ordered top to bottom — "
                "or switch to the sample funnel.")
        st.stop()
    raw = read_csv(upload)
    if raw is None:
        st.stop()
    editor_key = f"funnel-upload-{upload.name}-{upload.size}"

st.sidebar.caption("Edit counts, add or delete stages below.")
edited = st.sidebar.data_editor(
    raw, key=editor_key, num_rows="dynamic", hide_index=True, width="stretch",
    column_config={
        "stage": st.column_config.TextColumn("stage", required=True),
        "users": st.column_config.NumberColumn("users", min_value=0, step=1, format="%d"),
    },
)

st.sidebar.divider()
customer_value = st.sidebar.number_input(
    "Average customer value ($)", min_value=0.0, value=1200.0, step=100.0)
default_lift = st.sidebar.number_input(
    "Default relative lift per stage (%)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
leak_by = st.sidebar.selectbox(
    "Define 'largest leakage' by", list(LEAKAGE_METRICS), format_func=LEAKAGE_METRICS.get)
st.sidebar.caption("Synthetic data only — for demonstration.")

# ------------------------------------------------------------------ validation

st.title("Funnel Friction Analyzer")
st.caption("Signal → Diagnosis → Priority → Action: find where the funnel leaks and "
           "which fix is worth doing first.")

validation = validate_funnel(edited)
if not validation.ok:
    st.error("**The funnel data has problems that need fixing before analysis:**")
    for msg in validation.errors:
        st.markdown(f"- {msg}")
    st.stop()
for msg in validation.warnings:
    st.warning(msg)

funnel_df = clean_funnel(edited)
metrics = compute_metrics(funnel_df)
transitions = transition_table(funnel_df)
leak_idx = largest_leakage(funnel_df, leak_by)
leak = transitions.iloc[leak_idx]

# ------------------------------------------------------------------ KPI row

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Top of funnel · {metrics['stage'].iloc[0]}", f"{metrics['users'].iloc[0]:,.0f}")
k2.metric(f"Bottom · {metrics['stage'].iloc[-1]}", f"{metrics['users'].iloc[-1]:,.0f}")
k3.metric("End-to-end conversion", pct(metrics["cumulative_conversion"].iloc[-1], 2))
k4.metric("Largest leak · drop-off", pct(leak["drop_off"]),
          help=f"Leakage defined by: {LEAKAGE_METRICS[leak_by].lower()}")
k4.caption(f"{leak['transition']} · {leak['users_lost']:,.0f} users lost")

tab_diag, tab_prio = st.tabs(["① Diagnose", "② Prioritize"])

# ------------------------------------------------------------------ diagnose

with tab_diag:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Funnel")
        st.caption(f"Red bar (⚠) = destination of the largest leak: **{leak['transition']}**.")
        st.plotly_chart(funnel_chart(metrics, leak_idx + 1), width="stretch", theme="streamlit")
    with right:
        st.subheader("Stage metrics")
        st.dataframe(metrics_display(metrics), hide_index=True, width="stretch")
        st.markdown(
            f"**Largest leakage ({LEAKAGE_METRICS[leak_by].lower()}):** "
            f"{leak['transition']} converts {pct(leak['rate'])} — "
            f"{leak['users_lost']:,.0f} users ({pct(leak['drop_off'])}) drop out here."
        )

# ------------------------------------------------------------------ prioritize

with tab_prio:
    st.subheader("Assumptions per transition")
    st.caption("Lift is a *relative* improvement to that step's conversion rate "
               f"(25% → 27.5% is a 10% lift). Effort: {EFFORT_MIN} = trivial, "
               f"{EFFORT_MAX} = major project.")
    assumptions = pd.DataFrame({
        "transition": transitions["transition"],
        "current rate": transitions["rate"].map(pct),
        "lift %": default_lift,
        "effort": [SAMPLE_EFFORT.get(t, DEFAULT_EFFORT) for t in transitions["transition"]],
    })
    assumptions_key = f"assumptions-{hash(tuple(transitions['transition']))}-{default_lift}"
    assumptions = st.data_editor(
        assumptions, key=assumptions_key, hide_index=True, width="stretch",
        disabled=["transition", "current rate"],
        column_config={
            "lift %": st.column_config.NumberColumn(min_value=0.0, max_value=500.0, step=1.0,
                                                    format="%.1f%%"),
            "effort": st.column_config.NumberColumn(min_value=EFFORT_MIN, max_value=EFFORT_MAX,
                                                    step=1, format="%d"),
        },
    )
    lifts = assumptions["lift %"].fillna(0).clip(lower=0) / 100
    efforts = assumptions["effort"].fillna(DEFAULT_EFFORT).clip(EFFORT_MIN, EFFORT_MAX)

    opps = size_opportunities(funnel_df, lifts.tolist(), efforts.tolist(), customer_value)
    final_stage = metrics["stage"].iloc[-1]

    st.subheader("Ranked opportunities")
    table = ranked(opps)
    st.dataframe(
        pd.DataFrame({
            "Rank": table["rank"],
            "Transition": table["transition"],
            "Rate now → lifted": [f"{pct(a)} → {pct(b)}{' (capped)' if c else ''}"
                                  for a, b, c in zip(table["current_rate"], table["lifted_rate"],
                                                     table["capped"])],
            "+ next stage": table["incremental_next"].round(1),
            f"+ {final_stage}": table["incremental_final"].round(2),
            "+ Revenue ($)": table["incremental_revenue"].round(0),
            "Effort": table["effort"].astype(int),
            "Priority (customers / effort)": table["priority_score"].round(2),
        }),
        hide_index=True, width="stretch",
        column_config={"+ Revenue ($)": st.column_config.NumberColumn(format="dollar")},
    )
    st.download_button("Download ranked opportunities (CSV)", table.to_csv(index=False),
                       file_name="funnel_opportunities.csv", mime="text/csv")

    st.subheader("Summary")
    for line in build_summary(funnel_df, opps, customer_value, leak["transition"]):
        st.markdown(f"- {md_safe(line)}")
