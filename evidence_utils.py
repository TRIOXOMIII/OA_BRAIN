"""Filtering, parsing, and traceability helpers for workbook-backed evidence."""
from __future__ import annotations

import re
from urllib.parse import quote

import pandas as pd

from data_loader import clean_text


def options(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    return sorted({clean_text(value) for value in frame[column] if clean_text(value)}, key=str.casefold)


def apply_filters(findings: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    """Apply all sidebar controls consistently. Each condition is an exact workbook value."""
    result = findings.copy()
    if filters.get("resting_only"):
        result = result[result["_Is_Resting_State"]].copy()
    mappings = {
        "states": "EEG_State",
        "bands": "EEG_Band",
        "band_groups": "_Band_Group",
        "regions": "Brain_Region",
        "metrics": "EEG_Metric",
        "outcomes": "Clinical_Outcome",
        "domains": "_Clinical_Domain",
        "directions": "_Direction_Group",
        "analyses": "Analysis_Type",
    }
    for filter_name, column in mappings.items():
        selected = filters.get(filter_name, [])
        if selected:
            result = result[result[column].isin(selected)].copy()

    query = clean_text(filters.get("search", ""))
    if query:
        columns = [
            "Finding_ID", "Study_ID", "_Study_Title", "_Study_Authors", "EEG_Band",
            "Frequency_Hz", "Brain_Region", "EEG_Metric", "Clinical_Outcome",
            "Clinical_Measure", "Finding_Summary", "Authors_Interpretation",
            "Result_Direction", "Statistical_Test",
        ]
        haystack = result[[column for column in columns if column in result]].fillna("").astype(str)
        result = result[haystack.apply(lambda row: row.str.contains(re.escape(query), case=False, regex=True).any(), axis=1)].copy()
    return result


def source_url(value: object) -> str:
    value = clean_text(value)
    if not value:
        return ""
    if value.startswith(("https://", "http://")):
        return value
    if value.startswith("www."):
        return "https://" + value
    if value.casefold().startswith("doi:"):
        value = value[4:].strip()
    return "https://doi.org/" + quote(value, safe="/.:;()") if value.startswith("10.") else ""


def parse_exact_p_value(value: object) -> float | None:
    """Return only equality-form p-values; inequalities are not converted to false precision."""
    text = clean_text(value).replace(",", ".")
    match = re.fullmatch(r"p\s*=\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    return number if 0 <= number <= 1 else None


def parse_effect_and_ci(effect: object, ci: object) -> tuple[float, float, float] | None:
    """Parse only an unambiguous bracketed two-endpoint CI for a record-level display."""
    effect_text = clean_text(effect).replace("−", "-")
    ci_text = clean_text(ci).replace("−", "-")
    effect_match = re.search(r"(?:β|beta|b|r|rho)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+))", effect_text, re.IGNORECASE)
    ci_match = re.search(r"(?:\b\d{2}%\s*)?CI\s*\[\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*,\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*\]", ci_text, re.IGNORECASE)
    if not effect_match or not ci_match:
        return None
    estimate, lower, upper = map(float, (effect_match.group(1), ci_match.group(1), ci_match.group(2)))
    if lower <= estimate <= upper:
        return estimate, lower, upper
    return None


def coverage_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    total = len(frame)
    rows = []
    for column in columns:
        present = int(frame[column].map(lambda value: bool(clean_text(value))).sum()) if column in frame else 0
        rows.append({
            "Workbook field": column,
            "Records with a value": present,
            "Records in current view": total,
            "Coverage (%)": round(100 * present / total, 1) if total else 0.0,
        })
    return pd.DataFrame(rows)


def finding_trace(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Counts include Finding_IDs so every aggregate remains auditable."""
    clean_groups = [column for column in group_columns if column in frame]
    if frame.empty or not clean_groups:
        return pd.DataFrame()
    return (
        frame.groupby(clean_groups, dropna=False)
        .agg(
            Finding_records=("Finding_ID", "count"),
            Study_records=("Study_ID", "nunique"),
            Finding_IDs=("Finding_ID", lambda series: ", ".join(sorted({clean_text(v) for v in series if clean_text(v)}))),
        )
        .reset_index()
    )
