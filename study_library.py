"""Study-centred research library with progressive, source-linked exploration."""
from __future__ import annotations

import html
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from config import PALETTE
from data_loader import clean_text
from utils.evidence_utils import source_url
from utils.visualization_utils import theme_figure


def _safe(value: object) -> str:
    return html.escape(clean_text(value) or "Not recorded")


def _information_block(label: str, value: object) -> None:
    st.markdown(
        f"<div class='detail-block'><div class='detail-label'>{html.escape(label)}</div>"
        f"<div class='detail-value'>{_safe(value)}</div></div>",
        unsafe_allow_html=True,
    )


def _tags(values: list[str]) -> None:
    visible = [clean_text(value) for value in values if clean_text(value)]
    if not visible:
        st.caption("No value recorded for this source.")
        return
    chips = "".join(f"<span>{html.escape(value)}</span>" for value in visible)
    st.markdown(f"<div class='tag-list'>{chips}</div>", unsafe_allow_html=True)


def _select_study(study_id: str) -> None:
    st.session_state["library_study_id"] = study_id


def _study_cards(view: pd.DataFrame) -> str:
    candidate_ids = view["Study_ID"].tolist()
    if "library_study_id" not in st.session_state or st.session_state["library_study_id"] not in candidate_ids:
        st.session_state["library_study_id"] = candidate_ids[0]
    selected_id = st.session_state["library_study_id"]
    st.markdown("<div class='section-heading compact'><h2>Select a source study</h2><p>Each card opens a structured research view.</p></div>", unsafe_allow_html=True)
    columns = st.columns(2)
    for index, (_, study) in enumerate(view.iterrows()):
        title = clean_text(study["Full_Title"]) or "Untitled study"
        with columns[index % 2]:
            selected = study["Study_ID"] == selected_id
            st.markdown(
                f"<div class='study-card {'selected' if selected else ''}'>"
                f"<div class='card-meta'>{_safe(study['Study_ID'])} · {_safe(study['Year'])}</div>"
                f"<h3>{html.escape(title)}</h3><p>{_safe(study['Evidence_Category'])} · {_safe(study['Study_Design'])}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Viewing this study" if selected else "Open study",
                key=f"study_card_{study['Study_ID']}",
                disabled=selected,
                on_click=_select_study,
                args=(study["Study_ID"],),
                width="stretch",
            )
    return selected_id


def _main_findings(related: pd.DataFrame) -> None:
    if related.empty:
        st.info("No linked evidence observations are available for this study.", icon="ℹ️")
        return
    left, right = st.columns(2)
    with left:
        counts = related.assign(_Band=related["EEG_Band"].map(lambda value: clean_text(value) or "Not reported")).groupby("_Band").size().reset_index(name="Reported observations")
        figure = px.bar(counts.sort_values("Reported observations"), x="Reported observations", y="_Band", orientation="h", title="Reported EEG band labels", labels={"_Band": "EEG band"}, color_discrete_sequence=[PALETTE["cyan"]])
        st.plotly_chart(theme_figure(figure, 480), width="stretch")
    with right:
        counts = related.assign(_Outcome=related["Clinical_Outcome"].map(lambda value: clean_text(value) or "Not reported")).groupby("_Outcome").size().reset_index(name="Reported observations")
        figure = px.bar(counts.sort_values("Reported observations"), x="Reported observations", y="_Outcome", orientation="h", title="Clinical outcomes represented", labels={"_Outcome": "Clinical outcome"}, color_discrete_sequence=[PALETTE["violet"]])
        st.plotly_chart(theme_figure(figure, 480), width="stretch")
    st.markdown("#### Individual observations")
    for _, finding in related.iterrows():
        label = " · ".join(part for part in [clean_text(finding["Finding_ID"]), clean_text(finding["EEG_Band"]) or "EEG not specified", clean_text(finding["Clinical_Outcome"]) or "Outcome not specified"] if part)
        with st.expander(label, expanded=False):
            st.markdown(f"<div class='finding-summary'>{_safe(finding['Finding_Summary'])}</div>", unsafe_allow_html=True)
            details_left, details_right = st.columns(2)
            with details_left:
                _information_block("EEG metric", finding["EEG_Metric"])
                _information_block("Brain region", finding["Brain_Region"])
                _information_block("Recording state", finding["EEG_State"])
            with details_right:
                _information_block("Clinical measure", finding["Clinical_Measure"])
                _information_block("Reported direction", finding["Result_Direction"])
                _information_block("Statistical result", finding["P_Value"])


def _limitations(study: pd.Series, related: pd.DataFrame) -> None:
    st.markdown("<div class='limitation-card'><h3>Study-level considerations</h3><p>" + _safe(study["Main_Limitations"]) + "</p></div>", unsafe_allow_html=True)
    caveats = related[["Finding_ID", "Limitations_or_Caveats"]].copy()
    caveats = caveats[caveats["Limitations_or_Caveats"].map(lambda value: bool(clean_text(value)))]
    if caveats.empty:
        st.caption("No additional observation-level caveats are recorded.")
        return
    st.markdown("#### Observation-level considerations")
    cards = st.columns(2)
    for index, (_, caveat) in enumerate(caveats.iterrows()):
        with cards[index % 2]:
            st.markdown(f"<div class='caveat-card'><span>{_safe(caveat['Finding_ID'])}</span><p>{_safe(caveat['Limitations_or_Caveats'])}</p></div>", unsafe_allow_html=True)


def _compare_profiles(study: pd.Series, related: pd.DataFrame, studies: pd.DataFrame, findings: pd.DataFrame) -> None:
    """Compare recorded study profiles without treating unlike statistical results as comparable effects."""
    alternatives = studies[(studies["Study_ID"].ne("")) & (studies["Study_ID"] != study["Study_ID"])].drop_duplicates("Study_ID", keep="first")
    if alternatives.empty:
        st.caption("A second source is needed to compare research profiles.")
        return
    labels = {row["Study_ID"]: f"{row['Study_ID']} — {(clean_text(row['Full_Title']) or 'Untitled source')[:78]}" for _, row in alternatives.iterrows()}
    comparison_id = st.selectbox("Compare with another source", alternatives["Study_ID"].tolist(), format_func=lambda value: labels[value], key="study_profile_comparison")
    other = alternatives[alternatives["Study_ID"] == comparison_id].iloc[0]
    other_related = findings[findings["Study_ID"] == comparison_id].copy()
    left, right = st.columns(2)
    for column, title in ((left, clean_text(study["Study_ID"])), (right, clean_text(other["Study_ID"]))):
        source = study if title == clean_text(study["Study_ID"]) else other
        with column:
            st.markdown(f"<div class='comparison-heading'>{_safe(title)}<span>{_safe(source['Year'])}</span></div>", unsafe_allow_html=True)
            _information_block("Study design", source["Study_Design"])
            _information_block("Sample", source["Total_N"])
            _information_block("EEG state", source["EEG_State"])
            _information_block("Peer-review status", source["Peer_Review_Status"])
    frames = []
    for label, source_frame in ((clean_text(study["Study_ID"]), related), (clean_text(other["Study_ID"]), other_related)):
        grouped = source_frame.assign(_Band=source_frame["EEG_Band"].map(lambda value: clean_text(value) or "Not reported")).groupby("_Band").size().reset_index(name="Reported observations")
        grouped["Source study"] = label
        frames.append(grouped)
    chart_data = pd.concat(frames, ignore_index=True)
    figure = px.bar(chart_data, x="_Band", y="Reported observations", color="Source study", barmode="group", title="Reported EEG band labels by source", labels={"_Band": "EEG band"}, color_discrete_sequence=[PALETTE["cyan"], PALETTE["violet"]])
    st.plotly_chart(theme_figure(figure, 500), width="stretch")


def render(studies: pd.DataFrame, findings: pd.DataFrame) -> None:
    st.markdown("<div class='page-title'><div class='eyebrow'>SOURCE LIBRARY</div><h1>Research <span>Library</span></h1><p>Select a source to move from its broad research context to methodology, observations and considerations.</p></div>", unsafe_allow_html=True)
    query = st.text_input("Search sources", placeholder="Title, author, year or reference ID", key="library_search")
    view = studies[studies["Study_ID"].ne("")].drop_duplicates("Study_ID", keep="first").copy()
    if query.strip():
        fields = ["Study_ID", "Full_Title", "Authors", "Year", "Evidence_Category", "Study_Design", "Peer_Review_Status"]
        match = view[fields].fillna("").astype(str).apply(lambda row: row.str.contains(re.escape(query.strip()), case=False, regex=True).any(), axis=1)
        view = view[match].copy()
    if view.empty:
        st.info("No source matches that search.", icon="ℹ️")
        return
    selected_id = _study_cards(view)
    study = view[view["Study_ID"] == selected_id].iloc[0]
    related = findings[findings["Study_ID"] == selected_id].copy()

    st.markdown("<div class='study-detail-header'><div class='card-meta'>" + _safe(study["Study_ID"]) + " · " + _safe(study["Year"]) + "</div><h2>" + _safe(study["Full_Title"]) + "</h2><p>" + _safe(study["Authors"]) + "</p></div>", unsafe_allow_html=True)
    url = source_url(study.get("DOI_URL", ""))
    if url:
        st.link_button("Open source / DOI", url)

    tabs = st.tabs(["Overview", "Methodology", "Population & sample", "Main findings", "Limitations", "Relevant variables"])
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            _information_block("Research category", study["Evidence_Category"])
            _information_block("Research objective", study["Research_Objective"])
        with right:
            _information_block("Peer-review status", study["Peer_Review_Status"])
            _information_block("Reference ID", study["Study_ID"])
    with tabs[1]:
        left, right = st.columns(2)
        with left:
            _information_block("Study design", study["Study_Design"])
            _information_block("EEG recording state", study["EEG_State"])
        with right:
            _information_block("Recording conditions", study["EEG_Recording_Conditions"])
            _information_block("Clinical measures", study["Clinical_Measures"])
    with tabs[2]:
        left, right = st.columns(2)
        with left:
            _information_block("Population", study["Population_Description"])
            _information_block("Sample", study["Total_N"])
        with right:
            _information_block("Pain duration", study["Pain_Duration"])
            _information_block("Diagnostic phenotype", study["Diagnostic_Phenotypes"])
            _information_block("Phenotype definition", study["Phenotype_Definition"])
    with tabs[3]:
        _main_findings(related)
    with tabs[4]:
        _limitations(study, related)
    with tabs[5]:
        st.markdown("#### EEG band labels")
        _tags(sorted({clean_text(value) for value in related["EEG_Band"] if clean_text(value)}))
        st.markdown("#### Clinical outcomes")
        _tags(sorted({clean_text(value) for value in related["Clinical_Outcome"] if clean_text(value)}))
        st.markdown("#### EEG metrics")
        _tags(sorted({clean_text(value) for value in related["EEG_Metric"] if clean_text(value)}))
        st.markdown("#### Brain regions")
        _tags(sorted({clean_text(value) for value in related["Brain_Region"] if clean_text(value)}))
    with st.expander("Compare research profiles", expanded=False):
        _compare_profiles(study, related, studies, findings)
