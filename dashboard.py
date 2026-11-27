"""OA-BRAIN landing page."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import BAND_COLORS, PALETTE
from utils.evidence_utils import options
from utils.visualization_utils import empty_chart_message, theme_figure


def _metric(label: str, value: int, caption: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-caption'>{caption}</div></div>",
        unsafe_allow_html=True,
    )


def _count(frame: pd.DataFrame, column: str) -> int:
    return len(options(frame, column))


def render(studies: pd.DataFrame, findings: pd.DataFrame, set_page) -> None:
    st.markdown(
        """
        <section class='hero-panel'>
          <div class='eyebrow'>EVIDENCE INTELLIGENCE · OSTEOARTHRITIS</div>
          <h1>Map the signal.<br><span>Keep the evidence visible.</span></h1>
          <p>OA-BRAIN is an interactive research environment for exploring reported neurophysiological
          evidence in osteoarthritis. The current collection is centred on knee osteoarthritis and can grow
          with future evidence across the wider condition.</p>
          <div class='hero-pills'><span>Traceable evidence</span><span>Study-level exploration</span><span>EEG × clinical evidence</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-kicker'>EVIDENCE LANDSCAPE</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _metric("Research studies", int(studies["Study_ID"].ne("").sum()), "Sources represented")
    with m2:
        _metric("Evidence observations", len(findings), "Reported observations")
    with m3:
        _metric("Reported EEG bands", _count(findings, "EEG_Band"), "Distinct recorded labels")
    with m4:
        _metric("Clinical outcomes", _count(findings, "Clinical_Outcome"), "Domains represented")

    st.markdown("<div class='section-heading'><h2>Evidence at a glance</h2><p>Follow the distribution of reported EEG and clinical observations across the current collection.</p></div>", unsafe_allow_html=True)
    left, right = st.columns((1, 1))
    with left:
        band_counts = findings["_Band_Group"].value_counts().rename_axis("EEG band label").reset_index(name="Evidence observations")
        if band_counts.empty:
            empty_chart_message("No EEG band values are available in the current collection.")
        else:
            fig = px.bar(
                band_counts.sort_values("Evidence observations"), x="Evidence observations", y="EEG band label", orientation="h",
                color="EEG band label", color_discrete_map=BAND_COLORS,
                title="Reported observations by EEG band label",
                hover_data={"Evidence observations": True, "EEG band label": False},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(theme_figure(fig, 490), width="stretch")
    with right:
        state_counts = findings["EEG_State"].replace("", "Not reported").value_counts().rename_axis("EEG state").reset_index(name="Evidence observations")
        fig = px.pie(
            state_counts, names="EEG state", values="Evidence observations", hole=.62,
            title="Recording states represented in evidence",
            color_discrete_sequence=[PALETTE["cyan"], PALETTE["violet"], PALETTE["coral"], PALETTE["gold"]],
        )
        fig.update_traces(textinfo="label+value", textposition="outside")
        st.plotly_chart(theme_figure(fig, 490), width="stretch")

    a, b = st.columns((1.25, .75))
    with a:
        flow = findings.copy()
        for column in ("Evidence_Category", "EEG_State", "_Band_Group"):
            flow[column] = flow[column].replace("", "Not reported")
        if flow.empty:
            empty_chart_message("No evidence observations are available for this overview.")
        else:
            fig = px.sunburst(
                flow, path=["Evidence_Category", "EEG_State", "_Band_Group"],
                title="Evidence map: category → EEG state → reported band label",
                color="_Band_Group", color_discrete_map=BAND_COLORS,
            )
            fig.update_traces(hovertemplate="%{label}<br>Evidence observations: %{value}<extra></extra>")
            st.plotly_chart(theme_figure(fig, 560), width="stretch")
    with b:
        clinical = findings["Clinical_Outcome"].replace("", "Not reported").value_counts().rename_axis("Clinical outcome").reset_index(name="Evidence observations")
        fig = px.bar(
            clinical.sort_values("Evidence observations"), x="Evidence observations", y="Clinical outcome", orientation="h",
            title="Clinical outcomes represented", color_discrete_sequence=[PALETTE["teal"]],
        )
        st.plotly_chart(theme_figure(fig, 560), width="stretch")

    st.markdown("<div class='section-heading'><h2>Choose a research route</h2><p>Begin with the evidence map, inspect source studies, or browse individual observations.</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for column, title, body, page, button in [
        (c1, "Evidence Explorer", "Visualize reported relationships and navigate patterns across the collection.", "Evidence Explorer", "Explore relationships"),
        (c2, "Study Library", "Select a source study and progressively inspect its design, population, findings and caveats.", "Study Library", "Browse studies"),
        (c3, "Evidence Records", "Filter, compare and inspect individual observations while keeping their source context close.", "Evidence Records", "Browse records"),
    ]:
        with column:
            st.markdown(f"<div class='route-card'><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)
            st.button(button, key=f"route_{page}", width="stretch", on_click=set_page, args=(page,))

