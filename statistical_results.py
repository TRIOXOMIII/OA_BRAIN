"""Reported statistical evidence with conservative, record-level visual parsing."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PALETTE
from data_loader import clean_text
from utils.evidence_utils import parse_effect_and_ci, parse_exact_p_value
from utils.visualization_utils import empty_chart_message, theme_figure


STAT_COLUMNS = [
    "Finding_ID", "Study_ID", "_Study_Title", "EEG_State", "EEG_Band", "Frequency_Hz",
    "Brain_Region", "EEG_Metric", "Clinical_Outcome", "Clinical_Measure", "Group_or_Contrast",
    "Analysis_Type", "Statistical_Test", "Result_Direction", "Effect_Estimate", "P_Value",
    "Confidence_Interval", "Other_stadistics",
]


def _parsed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["_Parsed"] = rows.apply(lambda row: parse_effect_and_ci(row["Effect_Estimate"], row["Confidence_Interval"]), axis=1)
    rows = rows[rows["_Parsed"].notna()].copy()
    if rows.empty:
        return rows
    rows[["_Estimate", "_CI_Lower", "_CI_Upper"]] = pd.DataFrame(rows["_Parsed"].tolist(), index=rows.index)
    return rows


def _evidence_landscape(findings: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        data = findings.copy()
        data["_Band"] = data["EEG_Band"].map(lambda value: clean_text(value) or "Not reported")
        data["_Outcome"] = data["Clinical_Outcome"].map(lambda value: clean_text(value) or "Not reported")
        counts = data.groupby(["_Band", "_Outcome"], dropna=False).size().reset_index(name="Reported observations")
        figure = px.bar(counts, x="_Band", y="Reported observations", color="_Outcome", barmode="stack", title="Clinical outcomes across reported EEG bands", labels={"_Band": "EEG band", "_Outcome": "Clinical outcome"}, color_discrete_sequence=[PALETTE["cyan"], PALETTE["violet"], PALETTE["coral"], PALETTE["gold"]])
        st.plotly_chart(theme_figure(figure, 520), width="stretch")
    with right:
        data = findings.copy()
        data["_Analysis"] = data["Analysis_Type"].map(lambda value: clean_text(value) or "Not reported")
        data["_Study"] = data["Study_ID"].map(lambda value: clean_text(value) or "Not reported")
        counts = data.groupby(["_Analysis", "_Study"], dropna=False).size().reset_index(name="Reported observations")
        figure = px.bar(counts, x="Reported observations", y="_Analysis", color="_Study", orientation="h", title="Analytical approaches represented by source", labels={"_Analysis": "Analysis approach", "_Study": "Source study"}, color_discrete_sequence=[PALETTE["teal"], PALETTE["violet"], PALETTE["coral"], PALETTE["gold"]])
        st.plotly_chart(theme_figure(figure, 520), width="stretch")


def _p_values(findings: pd.DataFrame) -> None:
    exact = findings.copy()
    exact["_P_Exact"] = exact["P_Value"].map(parse_exact_p_value)
    comparable = exact[exact["_P_Exact"].notna()].copy()
    st.caption("Only standalone equality-form p-values are charted. Inequalities and multi-value entries remain available in the verification view without being converted to a single value.")
    if comparable.empty:
        empty_chart_message("No equality-form p-values are available in this selection.")
        return
    figure = px.histogram(comparable, x="_P_Exact", nbins=min(12, max(4, len(comparable))), hover_data=["Finding_ID", "Study_ID", "P_Value", "EEG_Band", "Clinical_Measure"], title="Distribution of directly reported p-values", labels={"_P_Exact": "Reported p-value", "count": "Observations"}, color_discrete_sequence=[PALETTE["coral"]])
    figure.add_vline(x=.05, line_dash="dot", line_color=PALETTE["gold"], annotation_text="0.05 reference")
    st.plotly_chart(theme_figure(figure, 540), width="stretch")
    cards = st.columns(3)
    for index, (_, row) in enumerate(comparable.iterrows()):
        with cards[index % 3]:
            st.markdown(f"<div class='stat-card'><span>{clean_text(row['Finding_ID'])} · {clean_text(row['Study_ID'])}</span><strong>{clean_text(row['P_Value'])}</strong><p>{clean_text(row['EEG_Band']) or 'EEG band not recorded'}</p></div>", unsafe_allow_html=True)


def _estimates(findings: pd.DataFrame) -> None:
    parsed = _parsed_rows(findings)
    st.caption("Each point is an individual reported coefficient. A record appears only when its labeled estimate and bracketed confidence interval can be read unambiguously; values are not combined across studies.")
    if parsed.empty:
        empty_chart_message("No records with an unambiguous labeled estimate and bracketed confidence interval are available in this selection.")
        return
    parsed["_Label"] = parsed.apply(lambda row: f"{row['Finding_ID']} · {row['Study_ID']}", axis=1)
    parsed = parsed.sort_values("_Estimate")
    figure = go.Figure(go.Scatter(
        x=parsed["_Estimate"], y=parsed["_Label"], mode="markers", marker=dict(size=11, color=PALETTE["cyan"]),
        error_x=dict(type="data", symmetric=False, array=(parsed["_CI_Upper"] - parsed["_Estimate"]).tolist(), arrayminus=(parsed["_Estimate"] - parsed["_CI_Lower"]).tolist(), color=PALETTE["violet"]),
        customdata=parsed[["Effect_Estimate", "Confidence_Interval", "P_Value", "Clinical_Measure"]].to_numpy(),
        hovertemplate="%{y}<br>Parsed point estimate: %{x}<br>Original estimate: %{customdata[0]}<br>Original CI: %{customdata[1]}<br>Original p-value: %{customdata[2]}<br>Clinical measure: %{customdata[3]}<extra></extra>",
    ))
    figure.add_vline(x=0, line_dash="dot", line_color=PALETTE["gold"])
    figure.update_layout(title="Individual reported estimates and confidence intervals", xaxis_title="Reported coefficient", yaxis_title="Observation")
    st.plotly_chart(theme_figure(figure, max(520, 86 * len(parsed))), width="stretch")


def _verification_cards(table: pd.DataFrame) -> None:
    """Keep original values inspectable without making a spreadsheet grid the default UI."""
    for _, row in table.iterrows():
        label = " · ".join(part for part in [clean_text(row.get("Finding_ID", "")), clean_text(row.get("Study_ID", "")), clean_text(row.get("EEG_Band", ""))] if part)
        with st.expander(label or "Reported statistical observation", expanded=False):
            left, right = st.columns(2)
            with left:
                for column, label_text in [
                    ("Clinical_Outcome", "Clinical outcome"),
                    ("Clinical_Measure", "Clinical measure"),
                    ("Analysis_Type", "Analysis approach"),
                    ("Statistical_Test", "Statistical test"),
                ]:
                    st.markdown(f"<div class='detail-block'><div class='detail-label'>{label_text}</div><div class='detail-value'>{clean_text(row.get(column, '')) or 'Not recorded'}</div></div>", unsafe_allow_html=True)
            with right:
                for column, label_text in [
                    ("Effect_Estimate", "Effect estimate"),
                    ("P_Value", "Reported p-value"),
                    ("Confidence_Interval", "Confidence interval"),
                    ("Other_statistics", "Other statistics"),
                ]:
                    st.markdown(f"<div class='detail-block'><div class='detail-label'>{label_text}</div><div class='detail-value'>{clean_text(row.get(column, '')) or 'Not recorded'}</div></div>", unsafe_allow_html=True)


def render(findings: pd.DataFrame, set_page) -> None:
    st.markdown("<div class='page-title'><div class='eyebrow'>REPORTED RESULTS</div><h1>Statistical <span>Evidence</span></h1><p>Explore the analytical results recorded for each observation, keeping their original wording and study context available for verification.</p></div>", unsafe_allow_html=True)
    if findings.empty:
        empty_chart_message("No reported results match the active filters.")
        return
    table = findings[[column for column in STAT_COLUMNS if column in findings]].copy().rename(columns={"_Study_Title": "Study title", "Other_stadistics": "Other statistics"})
    c1, c2, c3 = st.columns(3)
    c1.metric("Reported observations", len(table))
    c2.metric("Source studies", table["Study_ID"].nunique())
    c3.metric("p-values recorded", int(table["P_Value"].map(lambda value: bool(clean_text(value))).sum()))
    tabs = st.tabs(["Evidence landscape", "Reported p-values", "Estimates & intervals"])
    with tabs[0]:
        _evidence_landscape(findings)
    with tabs[1]:
        _p_values(findings)
    with tabs[2]:
        _estimates(findings)
    with st.expander("Verify or export the reported statistical records", expanded=False):
        st.caption("Open an observation to review its original values and analytical context.")
        _verification_cards(table)
        st.download_button("Export reported results (CSV)", table.to_csv(index=False).encode("utf-8-sig"), "ao_brain_reported_results.csv", "text/csv")
    st.button("Open evidence records for complete verification", key="statistics_to_records", on_click=set_page, args=("Evidence Records",), width="content")
