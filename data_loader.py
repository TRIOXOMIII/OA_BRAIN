"""Single, defensive workbook loading path used by every OA-BRAIN module."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from config import (
    FINDING_COLUMNS,
    REQUIRED_FINDING_COLUMNS,
    REQUIRED_STUDY_COLUMNS,
    STUDY_COLUMNS,
)


@dataclass(frozen=True)
class EvidenceData:
    """Workbook records plus derived display fields; raw source columns are retained."""

    studies: pd.DataFrame
    findings: pd.DataFrame
    sheet_names: tuple[str, ...]
    unmatched_study_ids: tuple[str, ...]


def clean_text(value: object) -> str:
    """Return a display-safe, normalized text value without altering the Excel file."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.casefold() in {"nan", "none", "null", "<na>"} else text


def clean_frame(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Keep every supplied column and add absent optional columns as blank strings."""
    result = frame.copy()
    for column in expected_columns:
        if column not in result.columns:
            result[column] = ""
    for column in result.columns:
        result[column] = result[column].map(clean_text)
    return result


def _find_sheet(sheet_names: list[str], wanted: str) -> str | None:
    return next((name for name in sheet_names if name.casefold() == wanted.casefold()), None)


def classify_direction(value: object) -> str:
    """A display grouping only; Result_Direction remains unmodified in the data table."""
    text = clean_text(value).casefold()
    if not text:
        return "Not reported"
    if any(token in text for token in ("no significant", "not significant", "non-significant")):
        return "No significant association"
    if "mixed" in text:
        return "Mixed findings"
    if "negative" in text and "association" in text:
        return "Negative association"
    if "positive" in text and "association" in text:
        return "Positive association"
    return clean_text(value)


def classify_band(value: object) -> str:
    """Do not split reported multi-band or full-band findings into invented components."""
    label = clean_text(value)
    lower = label.casefold()
    if not label:
        return "Not reported"
    if "full band" in lower or "broadband" in lower:
        return "Broadband / full band"
    if "," in label or " across " in lower or " to gamma" in lower:
        return "Combined / multi-band"
    return label


def classify_domain(value: object) -> str:
    """Coarse navigation grouping based only on the existing Clinical_Outcome label."""
    text = clean_text(value).casefold()
    has_pain = "pain" in text
    has_function = "function" in text or "mobility" in text
    if has_pain and has_function:
        return "Pain and physical function"
    if has_pain:
        return "Pain"
    if has_function:
        return "Physical function / mobility"
    return clean_text(value) or "Not reported"


def is_resting_state(value: object) -> bool:
    """Recognize the recorded Resting-state label without treating task recordings as rest."""
    return bool(re.search(r"\bresting[ -]?state\b", clean_text(value), flags=re.IGNORECASE))


def _add_context(studies: pd.DataFrame, findings: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    unique_studies = studies[studies["Study_ID"].ne("")].drop_duplicates("Study_ID", keep="first")
    lookup = unique_studies.set_index("Study_ID", drop=False).to_dict(orient="index")
    result = findings.copy()
    for target, source in {
        "_Study_Title": "Full_Title",
        "_Study_Authors": "Authors",
        "_Study_Year": "Year",
        "_Study_DOI": "DOI_URL",
        "_Study_Limitations": "Main_Limitations",
    }.items():
        result[target] = result["Study_ID"].map(lambda sid: lookup.get(sid, {}).get(source, ""))
    result["_Direction_Group"] = result["Result_Direction"].map(classify_direction)
    result["_Band_Group"] = result["EEG_Band"].map(classify_band)
    result["_Clinical_Domain"] = result["Clinical_Outcome"].map(classify_domain)
    result["_Is_Resting_State"] = result["EEG_State"].map(is_resting_state)
    known_ids = set(unique_studies["Study_ID"])
    unmatched = tuple(sorted({sid for sid in result["Study_ID"] if sid and sid not in known_ids}))
    return result, unmatched


@st.cache_data(show_spinner=False)
def load_evidence(workbook_path: str, modified_time: float) -> EvidenceData:
    """Read the source once per workbook version and return normalized in-memory records."""
    del modified_time  # Cache invalidation key; the path is the actual input.
    path = Path(workbook_path)
    excel = pd.ExcelFile(path)
    study_sheet = _find_sheet(excel.sheet_names, "Studies")
    finding_sheet = _find_sheet(excel.sheet_names, "Findings")
    if not study_sheet or not finding_sheet:
        found = ", ".join(excel.sheet_names) or "none"
        raise ValueError(f"Required sheets 'Studies' and 'Findings' were not both found. Available sheets: {found}.")

    studies_raw = pd.read_excel(path, sheet_name=study_sheet)
    findings_raw = pd.read_excel(path, sheet_name=finding_sheet)
    missing_studies = [column for column in REQUIRED_STUDY_COLUMNS if column not in studies_raw.columns]
    missing_findings = [column for column in REQUIRED_FINDING_COLUMNS if column not in findings_raw.columns]
    if missing_studies or missing_findings:
        fragments = []
        if missing_studies:
            fragments.append("Studies: " + ", ".join(missing_studies))
        if missing_findings:
            fragments.append("Findings: " + ", ".join(missing_findings))
        raise ValueError("Missing required identifier columns — " + " | ".join(fragments))

    studies = clean_frame(studies_raw, STUDY_COLUMNS)
    findings = clean_frame(findings_raw, FINDING_COLUMNS)
    findings, unmatched = _add_context(studies, findings)
    return EvidenceData(studies, findings, tuple(excel.sheet_names), unmatched)
