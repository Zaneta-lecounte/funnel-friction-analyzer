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
    BLENDED,
    LEAKAGE_METRICS,
    clean_funnel,
    compute_metrics,
    largest_leakage,
    segment_comparison,
    segment_divergence,
    segment_insights,
    transition_table,
    validate_funnel,
    validate_segments,
)
from opportunity_model import (
    DEFAULT_EFFORT,
    EFFORT_MAX,
    EFFORT_MIN,
    benchmark_opportunity,
    benchmark_total_upside,
    build_summary,
    money,
    pluralize,
    ranked,
    size_opportunities,
)

ROOT = Path(__file__).parent
SAMPLE_FUNNEL = ROOT / "sample_funnel.csv"
SAMPLE_SEGMENTS = ROOT / "sample_segments.csv"

# Synthetic default effort (1 = easy, 5 = hard) for the sample funnel's transitions.
SAMPLE_EFFORT = {
    "Landing Page → Form Start": 1,
    "Form Start → Form Complete": 2,
    "Form Complete → Qualified Lead": 3,
    "Qualified Lead → Opportunity": 3,
    "Opportunity → Customer": 4,
}

# Illustrative (synthetic) target step-conversion rates for the sample funnel.
SAMPLE_BENCHMARKS = {
    "Landing Page → Form Start": 0.30,
    "Form Start → Form Complete": 0.65,
    "Form Complete → Qualified Lead": 0.40,
    "Qualified Lead → Opportunity": 0.35,
    "Opportunity → Customer": 0.25,
}

BAR_COLOR = "#2a78d6"
LEAK_COLOR = "#d03b3b"
# Categorical series colours, assigned in this fixed order (never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLENDED_COLOR = "#898781"

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


def segment_chart(comp: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    transitions = comp["transition"].drop_duplicates().tolist()
    segments = [x for x in comp["segment"].drop_duplicates() if x != BLENDED] + [BLENDED]
    # Horizontal groups draw bottom-up, so add traces in reverse to read top-down.
    for i, name in reversed(list(enumerate(segments))):
        part = comp[comp["segment"] == name]
        fig.add_trace(go.Bar(
            name=name, y=part["transition"], x=part["rate"], orientation="h",
            marker={"color": BLENDED_COLOR if name == BLENDED else SERIES_COLORS[i]},
            text=part["rate"].map(lambda v: f"{v:.1%}"), textposition="outside",
            hovertemplate=f"<b>{name}</b><br>%{{y}}<br>Step conversion: %{{x:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", bargap=0.25, bargroupgap=0.08, height=120 + 70 * len(transitions),
        xaxis={"tickformat": ".0%", "title": "Step conversion"},
        yaxis={"categoryorder": "array", "categoryarray": transitions[::-1]},
        legend={"orientation": "h", "y": 1.08, "x": 0, "traceorder": "reversed"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return fig


def benchmark_chart(bench: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Actual (red = below target)", y=bench["transition"], x=bench["actual_rate"], orientation="h",
        marker={"color": [LEAK_COLOR if s == "Below" else BAR_COLOR for s in bench["status"]]},
        text=[f"{a:.1%}" + (" ▼ below" if s == "Below" else "")
              for a, s in zip(bench["actual_rate"], bench["status"])],
        textposition="inside", insidetextanchor="start", textfont={"color": "#ffffff"},
        hovertemplate="<b>%{y}</b><br>Actual: %{x:.1%}<extra></extra>",
    ))
    has_target = bench["target_rate"].notna()
    fig.add_trace(go.Scatter(
        name="Target", y=bench.loc[has_target, "transition"],
        x=bench.loc[has_target, "target_rate"], mode="markers",
        marker={"symbol": "line-ns", "size": 26, "line": {"width": 3, "color": "#52514e"}},
        hovertemplate="<b>%{y}</b><br>Target: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        height=100 + 55 * len(bench), xaxis={"tickformat": ".0%", "title": "Step conversion"},
        yaxis={"categoryorder": "array", "categoryarray": bench["transition"].tolist()[::-1]},
        legend={"orientation": "h", "y": 1.1, "x": 0},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
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

tab_diag, tab_prio, tab_seg, tab_bench = st.tabs(
    ["① Diagnose", "② Prioritize", "③ Segments", "④ Benchmark"])

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

# ------------------------------------------------------------------ segments

with tab_seg:
    st.subheader("Segment comparison")
    st.caption("Long-format CSV with columns `segment, stage, users` — every segment lists the "
               "same stages in the same order (e.g. paid vs organic).")
    seg_source = st.radio("Segment data", ["Sample segments", "Upload CSV"], horizontal=True,
                          key="seg-source")
    seg_raw = pd.read_csv(SAMPLE_SEGMENTS)
    if seg_source == "Upload CSV":
        seg_upload = st.file_uploader("Segment CSV", type="csv", key="seg-upload")
        seg_raw = read_csv(seg_upload) if seg_upload is not None else None

    if seg_raw is None:
        st.info("Upload a segment CSV to compare funnels.")
    else:
        seg_validation = validate_segments(seg_raw)
        if not seg_validation.ok:
            st.error("**The segment data has problems:**")
            for msg in seg_validation.errors:
                st.markdown(f"- {md_safe(msg)}")
        else:
            for msg in seg_validation.warnings:
                st.warning(msg)
            comp = segment_comparison(seg_raw)
            st.plotly_chart(segment_chart(comp), width="stretch", theme="streamlit")

            st.markdown("**Where segments leak differently**")
            for line in segment_insights(seg_raw, leak_by):
                st.markdown(f"- {md_safe(line)}")

            div = segment_divergence(seg_raw)
            rate_cols = [c for c in div.columns if c.startswith("rate · ")]
            st.dataframe(
                div.assign(**{c: div[c].map(pct) for c in rate_cols},
                           spread_pp=div["spread_pp"].round(1)),
                hide_index=True, width="stretch",
                column_config={"transition": "Transition", "best_segment": "Best",
                               "worst_segment": "Worst", "spread_pp": "Spread (pp)"},
            )
            with st.expander("Full per-segment metrics"):
                st.dataframe(
                    comp.assign(rate=comp["rate"].map(pct), drop_off=comp["drop_off"].map(pct),
                                rate_vs_blended_pp=comp["rate_vs_blended_pp"].round(1)),
                    hide_index=True, width="stretch",
                )

# ------------------------------------------------------------------ benchmark

with tab_bench:
    st.subheader("Gap to benchmark")
    st.caption("Enter a target step-conversion rate per transition (leave blank for none). "
               "Sample targets are illustrative, not industry data.")
    targets_in = pd.DataFrame({
        "transition": transitions["transition"],
        "actual": transitions["rate"].map(pct),
        "target %": [SAMPLE_BENCHMARKS[t] * 100 if t in SAMPLE_BENCHMARKS else None
                     for t in transitions["transition"]],
    })
    targets_in["target %"] = targets_in["target %"].astype(float)
    targets_in = st.data_editor(
        targets_in, key=f"targets-{hash(tuple(transitions['transition']))}",
        hide_index=True, width="stretch", disabled=["transition", "actual"],
        column_config={"target %": st.column_config.NumberColumn(
            min_value=0.0, max_value=100.0, step=0.5, format="%.1f%%")},
    )
    targets = [None if pd.isna(v) else v / 100 for v in targets_in["target %"]]

    bench = benchmark_opportunity(funnel_df, targets, customer_value)
    total = benchmark_total_upside(funnel_df, targets)
    final_stage = metrics["stage"].iloc[-1]
    below = bench[bench["status"] == "Below"]

    b1, b2, b3 = st.columns(3)
    b1.metric("Steps below target", f"{len(below)} of {int((bench['status'] != 'No target').sum())}")
    b2.metric(f"+ {pluralize(final_stage)} if all targets hit", f"{total:,.1f}")
    b3.metric("Revenue upside (all targets)", money(total * customer_value))

    st.plotly_chart(benchmark_chart(bench), width="stretch", theme="streamlit")
    st.dataframe(
        pd.DataFrame({
            "Transition": bench["transition"],
            "Actual": bench["actual_rate"].map(pct),
            "Target": bench["target_rate"].map(pct),
            "Gap (pp)": bench["gap_pp"].round(1),
            "Status": bench["status"],
            "Users short at next stage": bench["users_gap"].clip(lower=0).round(0),
            f"+ {final_stage} if this step hits target": bench["incremental_final"].round(2),
            "+ Revenue ($)": bench["incremental_revenue"].round(0),
        }),
        hide_index=True, width="stretch",
        column_config={"+ Revenue ($)": st.column_config.NumberColumn(format="dollar")},
    )
    if len(below):
        worst = below.sort_values("incremental_final", ascending=False).iloc[0]
        st.markdown(md_safe(
            f"**Biggest benchmark gap by impact:** {worst['transition']} runs "
            f"{abs(worst['gap_pp']):.1f} pp below target; closing it alone adds "
            f"~{worst['incremental_final']:,.1f} {pluralize(final_stage)} "
            f"(~{money(worst['incremental_revenue'])}). Closing every gap compounds to "
            f"~{total:,.1f}."
        ))
    else:
        st.success("Every transition with a target is at or above it.")
