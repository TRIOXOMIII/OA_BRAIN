"""Interactive evidence-record browser with source context and optional raw verification."""
from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from config import DIRECTION_COLORS, FINDING_COLUMNS, PALETTE
from data_loader import clean_text
from utils.evidence_utils import source_url
from utils.visualization_utils import empty_chart_message, theme_figure


def _safe(value: object) -> str:
    return html.escape(clean_text(value) or "Not recorded")


def _open_source(study_id: str) -> None:
    st.session_state["library_study_id"] = study_id
    st.session_state["ao_page"] = "Study Library"


def _detail(label: str, value: object) -> None:
    st.markdown(f"<div class='detail-block'><div class='detail-label'>{html.escape(label)}</div><div class='detail-value'>{_safe(value)}</div></div>", unsafe_allow_html=True)


def _select_finding(finding_id: str) -> None:
    st.session_state["finding_selection"] = finding_id


def _finding_cards(data: pd.DataFrame) -> str:
    ids = data["Finding_ID"].tolist()
    if "finding_selection" not in st.session_state or st.session_state["finding_selection"] not in ids:
        st.session_state["finding_selection"] = ids[0]
    selected_id = st.session_state["finding_selection"]
    st.markdown("<div class='section-heading compact'><h2>Evidence observations</h2><p>Select an observation to inspect its reported context and source.</p></div>", unsafe_allow_html=True)
    columns = st.columns(2)
    for index, (_, row) in enumerate(data.iterrows()):
        title = clean_text(row["Finding_Summary"]) or "No summary recorded"
        with columns[index % 2]:
            selected = row["Finding_ID"] == selected_id
            st.markdown(
                f"<div class='finding-card {'selected' if selected else ''}'>"
                f"<div class='card-meta'>{_safe(row['Finding_ID'])} · {_safe(row['Study_ID'])}</div>"
                f"<h3>{_safe(row['EEG_Band'])} <span>×</span> {_safe(row['Clinical_Outcome'])}</h3>"
                f"<p>{html.escape(title)}</p></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Viewing observation" if selected else "Inspect observation",
                key=f"finding_card_{row['Finding_ID']}",
                disabled=selected,
                on_click=_select_finding,
                args=(row["Finding_ID"],),
                width="stretch",
            )
    return selected_id


def _visual_overview(data: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        chart = data.copy()
        chart["_Band"] = chart["EEG_Band"].map(lambda value: clean_text(value) or "Not reported")
        chart["_Direction"] = chart["_Direction_Group"].map(lambda value: clean_text(value) or "Not reported")
        counts = chart.groupby(["_Band", "_Direction"], dropna=False).size().reset_index(name="Reported observations")
        figure = px.bar(counts, x="_Band", y="Reported observations", color="_Direction", barmode="stack", title="Reported direction by EEG band label", labels={"_Band": "EEG band", "_Direction": "Reported direction"}, color_discrete_map=DIRECTION_COLORS)
        st.plotly_chart(theme_figure(figure, 510), width="stretch")
    with right:
        chart = data.copy()
        chart["_Metric"] = chart["EEG_Metric"].map(lambda value: clean_text(value) or "Not reported")
        chart["_Outcome"] = chart["Clinical_Outcome"].map(lambda value: clean_text(value) or "Not reported")
        counts = chart.groupby(["_Metric", "_Outcome"], dropna=False).size().reset_index(name="Reported observations")
        figure = px.sunburst(counts, path=["_Metric", "_Outcome"], values="Reported observations", title="EEG metrics and clinical outcomes", color="Reported observations", color_continuous_scale=[PALETTE["surface_2"], PALETTE["violet"], PALETTE["cyan"]])
        st.plotly_chart(theme_figure(figure, 510), width="stretch")


def _finding_detail(row: pd.Series, studies: pd.DataFrame) -> None:
    st.markdown("<div class='section-heading compact'><h2>Selected observation</h2><p>" + _safe(row["Finding_ID"]) + " · " + _safe(row["Study_ID"]) + "</p></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Reported observation", "EEG & clinical context", "Analysis", "Interpretation & caveats"])
    with tabs[0]:
        st.markdown(f"<div class='finding-focus'>{_safe(row['Finding_Summary'])}</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            _detail("Reference ID", row["Finding_ID"])
            _detail("Reported direction", row["Result_Direction"])
        with right:
            _detail("Source study", row["Study_ID"])
            _detail("Recording condition", row["Task_or_Condition"])
    with tabs[1]:
        left, right = st.columns(2)
        with left:
            _detail("EEG state", row["EEG_State"])
            _detail("EEG band", row["EEG_Band"])
            _detail("Frequency", row["Frequency_Hz"])
            _detail("Brain region", row["Brain_Region"])
        with right:
            _detail("EEG metric", row["EEG_Metric"])
            _detail("Clinical outcome", row["Clinical_Outcome"])
            _detail("Clinical measure", row["Clinical_Measure"])
            _detail("Group or contrast", row["Group_or_Contrast"])
    with tabs[2]:
        left, right = st.columns(2)
        with left:
            _detail("Analysis type", row["Analysis_Type"])
            _detail("Statistical test", row["Statistical_Test"])
            _detail("Effect estimate", row["Effect_Estimate"])
        with right:
            _detail("Reported p-value", row["P_Value"])
            _detail("Confidence interval", row["Confidence_Interval"])
            _detail("Other reported statistics", row["Other_stadistics"])
    with tabs[3]:
        st.markdown("<div class='interpretation-card'><h3>Authors’ interpretation</h3><p>" + _safe(row["Authors_Interpretation"]) + "</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='limitation-card'><h3>Considerations for this observation</h3><p>" + _safe(row["Limitations_or_Caveats"]) + "</p></div>", unsafe_allow_html=True)
    source = studies[studies["Study_ID"] == clean_text(row["Study_ID"])]
    if source.empty:
        return
    study = source.iloc[0]
    st.markdown("<div class='source-strip'><div><span>Source study</span><strong>" + _safe(study["Full_Title"]) + "</strong><small>" + _safe(study["Authors"]) + " · " + _safe(study["Year"]) + "</small></div></div>", unsafe_allow_html=True)
    buttons = st.columns(2)
    with buttons[0]:
        st.button("Open source study", key=f"open_source_{row['Finding_ID']}", on_click=_open_source, args=(clean_text(row["Study_ID"]),), width="stretch")
    url = source_url(study.get("DOI_URL", ""))
    if url:
        with buttons[1]:
            st.link_button("Open DOI / source", url, width="stretch")


def render(findings: pd.DataFrame, studies: pd.DataFrame) -> None:
    st.markdown("<div class='page-title'><div class='eyebrow'>EVIDENCE BROWSER</div><h1>Reported <span>Observations</span></h1><p>Filter the collection, compare its distributions, and progressively inspect each observation in its study context.</p></div>", unsafe_allow_html=True)
    original_columns = [column for column in FINDING_COLUMNS if column in findings]
    data = findings[original_columns].copy()
    if data.empty:
        empty_chart_message("No evidence observations match the active filters.")
        return
    first, second, third = st.columns(3)
    first.metric("Observations", len(data))
    second.metric("Source studies", data["Study_ID"].nunique())
    third.metric("Clinical outcomes", len({clean_text(value) for value in data["Clinical_Outcome"] if clean_text(value)}))
    _visual_overview(findings)
    selected_id = _finding_cards(data)
    selected = findings[findings["Finding_ID"] == selected_id].iloc[0]
    _finding_detail(selected, studies)
    with st.expander("Verify or export the original record set", expanded=False):
        st.caption("Use this view when you need to inspect the full extracted fields or export the active selection.")
        st.dataframe(data, width="stretch", hide_index=True, height=360)
        st.download_button("Export selected observations (CSV)", data.to_csv(index=False).encode("utf-8-sig"), "ao_brain_selected_observations.csv", "text/csv")
