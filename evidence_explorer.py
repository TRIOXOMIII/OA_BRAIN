"""Evidence Map, Reported Patterns, and Research Insights views."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import BAND_COLORS, DIRECTION_COLORS, PALETTE
from data_loader import clean_text
from utils.evidence_utils import coverage_table, finding_trace
from utils.visualization_utils import empty_chart_message, theme_figure


def _label(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(lambda value: clean_text(value) or "Not reported")


def _sankey(frame: pd.DataFrame) -> go.Figure:
    layers = [("_Band_Group", "EEG band"), ("EEG_Metric", "EEG metric"), ("Clinical_Outcome", "Clinical outcome")]
    records = frame.copy()
    nodes: list[str] = []
    index: dict[str, int] = {}
    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    for (left, left_title), (right, right_title) in zip(layers, layers[1:]):
        edges = pd.DataFrame({
            "source": _label(records, left).map(lambda x: f"{left_title}: {x}"),
            "target": _label(records, right).map(lambda x: f"{right_title}: {x}"),
        }).value_counts().reset_index(name="value")
        for _, edge in edges.iterrows():
            for node in (edge["source"], edge["target"]):
                if node not in index:
                    index[node] = len(nodes)
                    nodes.append(node)
            sources.append(index[edge["source"]])
            targets.append(index[edge["target"]])
            values.append(int(edge["value"]))
    return go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=16, thickness=16, label=nodes, color=PALETTE["surface_2"], line=dict(color=PALETTE["cyan"], width=.45)),
        link=dict(source=sources, target=targets, value=values, color="rgba(45,212,217,.28)", hovertemplate="%{source.label} → %{target.label}<br>Evidence observations: %{value}<extra></extra>"),
    ))


def _trace_cards(trace: pd.DataFrame) -> None:
    """Reveal aggregate provenance as expandable cards instead of a spreadsheet-style grid."""
    if trace.empty:
        st.caption("No traceable source groups are available for this selection.")
        return
    for _, row in trace.iterrows():
        band = clean_text(row.get("_Band_Group", "")) or "EEG band not reported"
        metric = clean_text(row.get("EEG_Metric", "")) or "EEG metric not reported"
        outcome = clean_text(row.get("Clinical_Outcome", "")) or "Clinical outcome not reported"
        with st.expander(f"{band} · {metric} · {outcome}", expanded=False):
            left, right = st.columns(2)
            with left:
                st.metric("Evidence observations", int(row["Finding_records"]))
            with right:
                st.metric("Source studies", int(row["Study_records"]))
            st.markdown(f"<div class='trace-ids'>{clean_text(row['Finding_IDs']) or 'No reference IDs recorded.'}</div>", unsafe_allow_html=True)


def _map_view(frame: pd.DataFrame) -> None:
    st.markdown("<div class='view-intro'><h2>Evidence Map</h2><p>Navigate the recorded links between EEG band labels, metrics and clinical outcomes. Multi-band and full-band rows remain intact as reported.</p></div>", unsafe_allow_html=True)
    if frame.empty:
        empty_chart_message("No evidence observations match the active filters. Adjust the selection to populate the map.")
        return
    fig = _sankey(frame)
    fig.update_layout(title="Reported evidence flow: EEG band → metric → clinical outcome")
    st.plotly_chart(theme_figure(fig, 640), width="stretch")

    left, right = st.columns(2)
    with left:
        cross = pd.crosstab(_label(frame, "_Band_Group"), _label(frame, "Clinical_Outcome"))
        if cross.empty:
            empty_chart_message("There are no band–outcome combinations to display.")
        else:
            heat = go.Figure(go.Heatmap(
                z=cross.values, x=cross.columns, y=cross.index, colorscale=[[0, "#0D2842"], [1, "#2DD4D9"]],
                hovertemplate="EEG band: %{y}<br>Clinical outcome: %{x}<br>Evidence observations: %{z}<extra></extra>",
            ))
            heat.update_layout(title="Evidence observations at each band–outcome intersection", xaxis_title="Clinical outcome", yaxis_title="Reported EEG band")
            st.plotly_chart(theme_figure(heat, 500), width="stretch")
    with right:
        region = finding_trace(frame, ["Brain_Region"])
        if region.empty:
            empty_chart_message("No brain-region values are recorded for the selected evidence.")
        else:
            region["Brain_Region"] = region["Brain_Region"].map(lambda value: clean_text(value) or "Not reported")
            fig = px.bar(region.sort_values("Finding_records"), x="Finding_records", y="Brain_Region", orientation="h", color="Study_records", color_continuous_scale=[PALETTE["violet"], PALETTE["cyan"]], title="Recorded brain regions")
            fig.update_traces(customdata=region[["Finding_IDs"]].to_numpy(), hovertemplate="%{y}<br>Evidence observations: %{x}<br>Source studies: %{marker.color}<br>Reference IDs: %{customdata[0]}<extra></extra>")
            st.plotly_chart(theme_figure(fig, 500), width="stretch")

    trace = finding_trace(frame, ["_Band_Group", "EEG_Metric", "Clinical_Outcome"])
    with st.expander("Inspect the traceable source records behind this map", expanded=False):
        _trace_cards(trace)


def _patterns_view(frame: pd.DataFrame, set_page) -> None:
    st.markdown("<div class='view-intro'><h2>Reported Patterns</h2><p>These charts organize the labels and directions reported in the articles. They do not synthesize an effect or infer a mechanism.</p></div>", unsafe_allow_html=True)
    if frame.empty:
        empty_chart_message("No evidence observations match the active filters.")
        return
    left, right = st.columns(2)
    with left:
        data = frame.copy()
        data["_Band"] = _label(data, "_Band_Group")
        data["_Direction"] = _label(data, "_Direction_Group")
        counts = data.groupby(["_Band", "_Direction"], dropna=False).size().reset_index(name="Evidence observations")
        fig = px.bar(counts, x="_Band", y="Evidence observations", color="_Direction", barmode="stack", color_discrete_map=DIRECTION_COLORS, title="Reported result directions by EEG band label", labels={"_Band": "Reported EEG band", "_Direction": "Reported direction"})
        st.plotly_chart(theme_figure(fig, 500), width="stretch")
    with right:
        metric = finding_trace(frame, ["EEG_Metric", "_Band_Group"])
        metric["EEG_Metric"] = metric["EEG_Metric"].map(lambda v: clean_text(v) or "Not reported")
        metric["_Band_Group"] = metric["_Band_Group"].map(lambda v: clean_text(v) or "Not reported")
        fig = px.treemap(metric, path=["EEG_Metric", "_Band_Group"], values="Finding_records", color="Finding_records", color_continuous_scale=[PALETTE["surface_2"], PALETTE["violet"], PALETTE["cyan"]], title="Reported metrics and band labels")
        fig.update_traces(customdata=metric[["Finding_IDs"]].to_numpy(), hovertemplate="%{label}<br>Evidence observations: %{value}<br>Reference IDs: %{customdata[0]}<extra></extra>")
        st.plotly_chart(theme_figure(fig, 500), width="stretch")
    with st.expander("Read the source observations supporting these patterns", expanded=False):
        st.caption("Reported observation text is kept separate from authors’ interpretations.")
        for _, row in frame.iterrows():
            label = " · ".join(part for part in [clean_text(row["Finding_ID"]), clean_text(row["Study_ID"]), clean_text(row["EEG_Band"])] if part)
            with st.expander(label or "Reported observation", expanded=False):
                st.markdown(f"<div class='finding-summary'>{clean_text(row['Finding_Summary']) or 'No reported observation text.'}</div>", unsafe_allow_html=True)
                st.caption("Clinical outcome: " + (clean_text(row["Clinical_Outcome"]) or "Not recorded") + " · Reported direction: " + (clean_text(row["Result_Direction"]) or "Not recorded"))
    st.button("Inspect reported statistical evidence", key="patterns_to_statistics", on_click=set_page, args=("Statistical Results",), width="content")


def _insights_view(frame: pd.DataFrame) -> None:
    st.markdown("<div class='view-intro'><h2>Research Insights</h2><p>A descriptive view of the evidence selected for exploration, including its clinical coverage and accompanying scientific context.</p></div>", unsafe_allow_html=True)
    if frame.empty:
        empty_chart_message("No evidence observations match the active filters.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Evidence observations", len(frame), "selected evidence")
    c2.metric("Source studies", frame["Study_ID"].nunique(), "represented")
    c3.metric("Reported p-values", int(frame["P_Value"].ne("").sum()), "available for review")

    left, right = st.columns((1.1, .9))
    with left:
        coverage = coverage_table(frame, ["Finding_Summary", "Authors_Interpretation", "Limitations_or_Caveats", "Effect_Estimate", "P_Value", "Confidence_Interval"])
        coverage["Evidence detail"] = coverage["Workbook field"].map({"Finding_Summary": "Reported observation", "Authors_Interpretation": "Authors’ interpretation", "Limitations_or_Caveats": "Scientific considerations", "Effect_Estimate": "Effect estimate", "P_Value": "p-value", "Confidence_Interval": "Confidence interval"})
        fig = px.bar(coverage, x="Coverage (%)", y="Evidence detail", orientation="h", text="Records with a value", range_x=[0, 100], title="Scientific detail available across the selected evidence", color_discrete_sequence=[PALETTE["teal"]])
        fig.update_traces(textposition="outside")
        st.plotly_chart(theme_figure(fig, 500), width="stretch")
    with right:
        observed = finding_trace(frame, ["_Clinical_Domain"])
        observed["_Clinical_Domain"] = observed["_Clinical_Domain"].map(lambda v: clean_text(v) or "Not reported")
        fig = px.bar(observed, x="_Clinical_Domain", y="Finding_records", color="Study_records", color_continuous_scale=[PALETTE["violet"], PALETTE["coral"]], title="Clinical domains represented in selected evidence", labels={"_Clinical_Domain": "Clinical domain"})
        fig.update_traces(customdata=observed[["Finding_IDs"]].to_numpy(), hovertemplate="%{x}<br>Evidence observations: %{y}<br>Source studies: %{marker.color}<br>Reference IDs: %{customdata[0]}<extra></extra>")
        st.plotly_chart(theme_figure(fig, 500), width="stretch")
    st.caption("Counts indicate representation in the selected evidence collection; open the source records to interpret each observation in context.")


def render(findings: pd.DataFrame, set_page) -> None:
    tabs = st.tabs(["Evidence Map", "Reported Patterns", "Research Insights"])
    with tabs[0]:
        _map_view(findings)
    with tabs[1]:
        _patterns_view(findings, set_page)
    with tabs[2]:
        _insights_view(findings)

