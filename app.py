"""OA-BRAIN: a biomedical evidence exploration platform."""
from __future__ import annotations

import streamlit as st

from config import ASSETS_DIR, WORKBOOK_PATH
from data_loader import load_evidence
from eeg_lab import render_eeg_lab
from modules import dashboard, evidence_explorer, findings_database, statistical_results, study_library
from utils.evidence_utils import apply_filters, options


st.set_page_config(page_title="OA-BRAIN · Evidence Intelligence", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")


def _load_css() -> None:
    style_path = ASSETS_DIR / "styles.css"
    if style_path.exists():
        st.markdown(f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _set_page(page: str) -> None:
    st.session_state["ao_page"] = page


def _sidebar(findings):
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'><div class='brain-mark'>◌</div><div><b>OA-BRAIN</b><span>Evidence intelligence</span></div></div>", unsafe_allow_html=True)
        pages = ["Home", "Evidence Explorer", "EEG Lab", "Statistical Results", "Study Library", "Evidence Records"]
        if "ao_page" not in st.session_state:
            st.session_state["ao_page"] = "Home"
        page = st.radio("Navigation", pages, key="ao_page", label_visibility="collapsed")
        st.markdown("<div class='filter-divider'></div><div class='sidebar-section'>EXPLORE EVIDENCE</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-footnote'><strong>EEG scope:</strong> Resting-state only. The current collection contains no task-based EEG recordings.</div>", unsafe_allow_html=True)
        filters = {
            "resting_only": False,
            "states": [],
            "bands": st.multiselect("Reported EEG band", options(findings, "EEG_Band")),
            "band_groups": st.multiselect("Band label type", options(findings, "_Band_Group"), help="Multi-band and full-band records remain distinct from individual bands."),
            "regions": st.multiselect("Brain region", options(findings, "Brain_Region")),
            "metrics": st.multiselect("EEG metric", options(findings, "EEG_Metric")),
            "outcomes": st.multiselect("Clinical outcome", options(findings, "Clinical_Outcome")),
            "domains": st.multiselect("Clinical domain", options(findings, "_Clinical_Domain")),
            "directions": st.multiselect("Reported direction", options(findings, "_Direction_Group")),
            "analyses": st.multiselect("Analysis type", options(findings, "Analysis_Type")),
            "search": st.text_input("Search evidence", placeholder="Reference ID, title, WOMAC…"),
        }
        st.markdown("<div class='sidebar-footnote'>Refine the evidence displayed across exploration, reported results, and individual records.</div>", unsafe_allow_html=True)
        st.markdown("<div class='filter-divider'></div>", unsafe_allow_html=True)
        st.download_button(
            "Download original dataset",
            data=WORKBOOK_PATH.read_bytes(),
            file_name=WORKBOOK_PATH.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            help="Download the source dataset for independent verification.",
        )
    return page, filters


def _render_page(page: str, evidence, filtered) -> None:
    renderers = {
        "Home": lambda: dashboard.render(evidence.studies, evidence.findings, _set_page),
        "Evidence Explorer": lambda: evidence_explorer.render(filtered, _set_page),
        "EEG Lab": lambda: render_eeg_lab(evidence.studies, evidence.findings, _set_page),
        "Statistical Results": lambda: statistical_results.render(filtered, _set_page),
        "Study Library": lambda: study_library.render(evidence.studies, evidence.findings),
        "Evidence Records": lambda: findings_database.render(filtered, evidence.studies),
    }
    try:
        renderers[page]()
    except Exception as error:  # A rendering failure remains isolated to its route.
        st.error(f"The {page} module could not be rendered. Other sections remain available.")
        st.exception(error)


def main() -> None:
    _load_css()
    if not WORKBOOK_PATH.exists():
        st.error(f"Evidence source not found: `{WORKBOOK_PATH.name}`. Keep it beside `app.py` and restart OA-BRAIN.")
        st.stop()
    try:
        evidence = load_evidence(str(WORKBOOK_PATH), WORKBOOK_PATH.stat().st_mtime)
    except Exception as error:
        st.error("OA-BRAIN could not load the evidence source. Check the source file and required identifiers.")
        st.exception(error)
        st.stop()
    page, filters = _sidebar(evidence.findings)
    filtered = apply_filters(evidence.findings, filters)
    if evidence.unmatched_study_ids:
        st.warning("Some Findings rows have Study_ID values not present in Studies: " + ", ".join(evidence.unmatched_study_ids))
    _render_page(page, evidence, filtered)


if __name__ == "__main__":
    main()
