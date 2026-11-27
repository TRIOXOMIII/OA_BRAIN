"""Shared configuration for the OA-BRAIN evidence platform."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKBOOK_PATH = BASE_DIR / "OA_NeuroEvidence_Template.xlsx"
ASSETS_DIR = BASE_DIR / "assets"

STUDY_COLUMNS = [
    "Study_ID", "Full_Title", "Authors", "Year", "DOI_URL", "Evidence_Category",
    "Study_Design", "Research_Objective", "Population_Description", "Total_N",
    "EEG_State", "EEG_Recording_Conditions", "Clinical_Measures", "Pain_Duration",
    "Diagnostic_Phenotypes", "Phenotype_Definition", "Main_Limitations",
    "Peer_Review_Status",
]

FINDING_COLUMNS = [
    "Finding_ID", "Study_ID", "Evidence_Category", "EEG_State", "Task_or_Condition",
    "EEG_Band", "Frequency_Hz", "Brain_Region", "EEG_Metric", "Clinical_Outcome",
    "Clinical_Measure", "Group_or_Contrast", "Analysis_Type", "Statistical_Test",
    "Result_Direction", "Effect_Estimate", "P_Value", "Confidence_Interval",
    "Other_stadistics", "Finding_Summary", "Authors_Interpretation",
    "Limitations_or_Caveats",
]

REQUIRED_STUDY_COLUMNS = ("Study_ID",)
REQUIRED_FINDING_COLUMNS = ("Finding_ID", "Study_ID")

PALETTE = {
    "navy": "#071A2F",
    "surface": "#0D2842",
    "surface_2": "#113553",
    "cyan": "#2DD4D9",
    "teal": "#20B9A8",
    "violet": "#A78BFA",
    "coral": "#FB8B7A",
    "gold": "#F4C95D",
    "text": "#F8FCFF",
    "muted": "#F8FCFF",
    "line": "#23445C",
}

BAND_COLORS = {
    "Delta": "#67B7DC",
    "Theta": "#A78BFA",
    "Alpha": "#2DD4D9",
    "Beta": "#F4C95D",
    "High beta": "#FB8B7A",
    "Gamma": "#EC6B9A",
    "Infraslow frequency (ISF)": "#6EE7B7",
    "Combined / multi-band": "#8198AA",
    "Broadband / full band": "#5E7790",
    "Not reported": "#718096",
}

DIRECTION_COLORS = {
    "Positive association": "#2DD4D9",
    "Negative association": "#FB8B7A",
    "Mixed findings": "#A78BFA",
    "No significant association": "#8198AA",
    "Not reported": "#718096",
}
