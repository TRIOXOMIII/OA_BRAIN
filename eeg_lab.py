"""Integrated OA-BRAIN EEG Lab interface.

This module connects literature context with reproducible synthetic signal processing.
It deliberately does not claim that its generated signals are participant recordings.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PALETTE
from data_loader import clean_text
from eeg_processing import BANDS, CHANNELS, DEFAULT_DURATION_SECONDS, SimulationParameters, analyse_raw, comparison_frame, generate_synthetic_raw
from neuro_explorer import BAND_COLORS, band_power_bar, channel_band_heatmap, electrode_3d_figure, psd_comparison_figure, time_series_figure, topomap_figure
from utils.evidence_utils import source_url
from utils.visualization_utils import empty_chart_message, theme_figure


@dataclass(frozen=True)
class EvidencePreset:
    preset_id: str
    study_id: str
    finding_ids: tuple[str, ...]
    label: str
    focus_band: str
    focus_channels: tuple[str, ...]
    purpose: str
    simulation_note: str


# These mappings deliberately contain no study-derived amplitude or patient-level values.
# They only route documented condition, band, and region information into the exploration UI.
PRESETS: tuple[EvidencePreset, ...] = (
    EvidencePreset(
        "s01_delta",
        "S01",
        ("F01.01", "F01.02"),
        "Delta exploration",
        "Delta",
        ("F3", "F4", "C3", "C4"),
        "Explore the recorded frontal and central delta-band context during eyes-closed resting EEG.",
        "The publication reports relative-power associations, not electrode amplitudes. The signal amplitude remains an illustrative computational input.",
    ),
    EvidencePreset(
        "s03_theta",
        "S03",
        ("F03.01", "F03.04"),
        "Central theta exploration",
        "Theta",
        ("C3", "C4"),
        "Explore the documented central theta context during eyes-closed resting EEG.",
        "The publication reports regression associations, not a waveform or amplitude profile. The signal amplitude remains an illustrative computational input.",
    ),
    EvidencePreset(
        "s04_beta",
        "S04",
        ("F04.03", "F04.05"),
        "Frontal beta exploration",
        "Beta",
        ("F3", "F4"),
        "Explore reported beta-band observations in the rehabilitation study's resting-state context.",
        "The publication reports subgroup findings without a numerical scalp-amplitude preset. The signal amplitude remains an illustrative computational input.",
    ),
)

ELECTRODE_GUIDE = {
    "Fp1": ("Frontopolar left", "Anterior left frontopolar position in the 10–20 montage."),
    "Fp2": ("Frontopolar right", "Anterior right frontopolar position in the 10–20 montage."),
    "F3": ("Frontal left", "Left frontal scalp position in the 10–20 montage."),
    "F4": ("Frontal right", "Right frontal scalp position in the 10–20 montage."),
    "C3": ("Central left", "Left central scalp position in the 10–20 montage."),
    "C4": ("Central right", "Right central scalp position in the 10–20 montage."),
    "O1": ("Occipital left", "Left occipital scalp position in the 10–20 montage."),
    "O2": ("Occipital right", "Right occipital scalp position in the 10–20 montage."),
}


def _safe(value: object) -> str:
    return html.escape(clean_text(value) or "Not recorded")


def _available_presets(studies: pd.DataFrame, findings: pd.DataFrame) -> list[EvidencePreset]:
    study_ids = set(studies["Study_ID"].map(clean_text))
    finding_ids = set(findings["Finding_ID"].map(clean_text))
    return [preset for preset in PRESETS if preset.study_id in study_ids and set(preset.finding_ids).issubset(finding_ids)]


def _select_preset(key: str, preset_id: str) -> None:
    st.session_state[key] = preset_id


def _preset_cards(presets: list[EvidencePreset], selected_id: str, state_key: str, heading: str) -> None:
    st.markdown(f"<div class='section-heading compact'><h2>{heading}</h2><p>Each configuration is guided by the documented study condition, band and regional context. It does not insert an unreported amplitude into the simulation.</p></div>", unsafe_allow_html=True)
    columns = st.columns(3)
    for index, preset in enumerate(presets):
        with columns[index % 3]:
            selected = preset.preset_id == selected_id
            st.markdown(
                f"<div class='eeg-preset-card {'selected' if selected else ''}'>"
                f"<div class='card-meta'>Literature-informed · {preset.focus_band}</div>"
                f"<h3>{_safe(preset.label)}</h3><p>{_safe(preset.purpose)}</p></div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Selected" if selected else "Select configuration",
                key=f"{state_key}_{preset.preset_id}",
                disabled=selected,
                on_click=_select_preset,
                args=(state_key, preset.preset_id),
                width="stretch",
            )


def _preset_from_id(presets: list[EvidencePreset], preset_id: str) -> EvidencePreset:
    return next(preset for preset in presets if preset.preset_id == preset_id)


def _reset_controls() -> None:
    keys = [
        "eeg_duration", "eeg_seed", "eeg_noise_a", "eeg_noise_b", "eeg_focus_gain_a", "eeg_focus_gain_b",
        "eeg_extra_band", "eeg_extra_gain", "eeg_channels", "eeg_window", "eeg_spatial_band", "eeg_measure",
        "eeg_selected_electrode", "eeg_config_b",
    ]
    for key in keys:
        st.session_state.pop(key, None)


def _build_parameters(preset: EvidencePreset, duration: float, seed: int, noise_uv: float, focus_gain: float, extra_band: str, extra_gain: float) -> SimulationParameters:
    scales = {extra_band: extra_gain} if extra_band != preset.focus_band else {}
    if extra_band == preset.focus_band:
        focus_gain *= extra_gain
    return SimulationParameters(
        duration_seconds=duration,
        seed=seed,
        noise_uv=noise_uv,
        focus_band=preset.focus_band,
        focus_channels=preset.focus_channels,
        focus_gain=focus_gain,
        band_scales=scales,
    )


def _open_research_library(study_id: str, set_page) -> None:
    """Route to the selected primary source without exposing an internal ID in EEG Lab."""
    st.session_state["library_study_id"] = study_id
    set_page("Study Library")


def _open_evidence_records(set_page) -> None:
    set_page("Evidence Records")


def _unique_values(frame: pd.DataFrame, column: str) -> str:
    values = [clean_text(value) for value in frame[column] if clean_text(value)]
    return " · ".join(dict.fromkeys(values)) or "Not recorded"


def _configuration_provenance(preset: EvidencePreset, studies: pd.DataFrame, findings: pd.DataFrame, set_page) -> None:
    """Keep source provenance useful, while leaving record-level claims to their own modules."""
    study = studies[studies["Study_ID"] == preset.study_id]
    related = findings[findings["Finding_ID"].isin(preset.finding_ids)]
    if study.empty:
        return
    source = study.iloc[0]
    documented_condition = _unique_values(related, "Task_or_Condition") if not related.empty else "Not recorded"
    documented_metric = _unique_values(related, "EEG_Metric") if not related.empty else "Not recorded"
    documented_region = _unique_values(related, "Brain_Region") if not related.empty else "Not recorded"
    st.markdown(
        "<div class='eeg-evidence-panel'><div class='card-meta'>Configuration provenance</div>"
        f"<h3>{_safe(preset.label)}</h3><p>This exploration is anchored to a documented resting-EEG condition, "
        "frequency-band context and regional description. Record-level results, statistics and caveats remain in the evidence modules, "
        "where they can be inspected in their complete source context.</p></div>",
        unsafe_allow_html=True,
    )
    source_columns = st.columns(2)
    with source_columns[0]:
        st.markdown(
            "<div class='detail-block'><div class='detail-label'>Documented source</div>"
            f"<div class='detail-value'>{_safe(source['Full_Title'])}<br>{_safe(source['Authors'])} · {_safe(source['Year'])}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='detail-block'><div class='detail-label'>Recording context used for orientation</div>"
            f"<div class='detail-value'>{_safe(source['EEG_State'])}<br>{_safe(source['EEG_Recording_Conditions'])}</div></div>",
            unsafe_allow_html=True,
        )
    with source_columns[1]:
        st.markdown(
            "<div class='detail-block'><div class='detail-label'>Fields informing this configuration</div>"
            f"<div class='detail-value'>Condition: {_safe(documented_condition)}<br>Region: {_safe(documented_region)}<br>Reported metric: {_safe(documented_metric)}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='detail-block'><div class='detail-label'>Configuration translation</div>"
            f"<div class='detail-value'>{_safe(preset.focus_band)} focus at {', '.join(preset.focus_channels)}. "
            "This selects an exploration target; it is not a source-derived amplitude estimate.</div></div>",
            unsafe_allow_html=True,
        )
    url = source_url(source.get("DOI_URL", ""))
    actions = st.columns(3)
    with actions[0]:
        st.button("Open Research Library", key="eeg_open_library", on_click=_open_research_library, args=(preset.study_id, set_page), width="stretch")
    with actions[1]:
        st.button("Open evidence records", key="eeg_open_records", on_click=_open_evidence_records, args=(set_page,), width="stretch")
    with actions[2]:
        if url:
            st.link_button("Open source / DOI", url, width="stretch")

    with st.expander("Configuration details and interpretation boundary", expanded=False):
        st.markdown(
            "<div class='eeg-configuration-grid'>"
            "<div><span>Source-informed inputs</span><strong>Condition, band and regional context determine the configuration category and focus channels.</strong></div>"
            "<div><span>Model inputs</span><strong>Sampling rate, duration, seed, noise and multipliers define a reproducible computational signal.</strong></div>"
            "<div><span>Calculated outputs</span><strong>Traces, Welch PSD, band power, topographies and B − A differences are computed from that signal.</strong></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='eeg-context-note'>The workbook contains published summaries rather than raw recordings. "
            "EEG Lab therefore supports transparent method exploration and sensitivity checks; it does not convert the reported evidence into participant-level signals, source-localization estimates or new clinical effects.</div>",
            unsafe_allow_html=True,
        )


def _difference_heatmap(difference: pd.DataFrame, measure: str) -> go.Figure:
    value_column = "Difference (B − A)"
    matrix = difference.pivot(index="Channel", columns="Band", values=value_column).reindex(columns=list(BANDS))
    max_abs = float(np.nanmax(np.abs(matrix.values))) if matrix.size else 1.0
    max_abs = max(max_abs, 1e-12)
    figure = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale="RdBu",
        reversescale=True,
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="B − A"),
        hovertemplate="Channel: %{y}<br>Band: %{x}<br>Calculated difference: %{z:.5g}<extra></extra>",
    ))
    unit = "relative power (%)" if measure == "relative" else "absolute power (V²)"
    figure.update_layout(title=f"Channel-by-band difference in {unit}", xaxis_title="Frequency band", yaxis_title="Channel", height=520)
    return theme_figure(figure)


def _comparison_summary(difference: pd.DataFrame) -> None:
    mean_difference = difference.groupby("Band", as_index=False)["Difference (B − A)"].mean().set_index("Band").reindex(list(BANDS)).reset_index()
    cards = st.columns(len(BANDS))
    for index, row in mean_difference.iterrows():
        with cards[index]:
            st.markdown(f"<div class='eeg-difference-card'><span>{_safe(row['Band'])}</span><strong>{float(row['Difference (B − A)']):+.3g}</strong><p>mean B − A</p></div>", unsafe_allow_html=True)


def render_eeg_lab(studies: pd.DataFrame, findings: pd.DataFrame, set_page) -> None:
    """Render the integrated literature-informed synthetic EEG laboratory."""
    presets = _available_presets(studies, findings)
    st.markdown("<div class='page-title eeg-title'><div class='eyebrow'>NEUROPHYSIOLOGICAL EXPLORATION</div><h1>EEG <span>Lab</span></h1><p>A reproducible computational workspace for inspecting signal behaviour, spectral structure and electrode-level displays in relation to the published osteoarthritis EEG evidence available in OA-BRAIN.</p></div>", unsafe_allow_html=True)
    if not presets:
        empty_chart_message("No evidence-guided EEG configurations can be created from the current source collection.")
        return
    st.markdown(
        "<div class='eeg-intro-grid'>"
        "<div><span>Purpose</span><strong>Test analysis choices on a controlled multichannel signal before transferring an analytic rationale to an external EEG dataset.</strong></div>"
        "<div><span>Functions</span><strong>Inspect time traces, compare Welch spectra, quantify band power, review scalp interpolation and contrast two configurations.</strong></div>"
        "<div><span>Working scope</span><strong>Signals are generated reproducibly from explicit model settings; published fields orient the configuration without being recast as measurements.</strong></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='eeg-researcher-guide'><span>Suggested analytical workflow</span>"
        "<strong>1.</strong> choose the evidence-informed category; <strong>2.</strong> inspect channels and time window before reading spectral output; "
        "<strong>3.</strong> hold the seed constant when attributing a B − A change to a selected multiplier; <strong>4.</strong> use the final provenance panel to open the original research context and evidence records.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='eeg-scope-note'><strong>Interpretation scope.</strong> The available workbook supplies curated research summaries, not raw participant recordings. "
        "This makes the Lab well suited to transparent computational exploration, parameter sensitivity and visual inspection, while keeping empirical inference anchored to the linked evidence and original publications.</div>",
        unsafe_allow_html=True,
    )
    if "eeg_config_a" not in st.session_state or st.session_state["eeg_config_a"] not in {preset.preset_id for preset in presets}:
        st.session_state["eeg_config_a"] = presets[0].preset_id
    selected_a_id = st.session_state["eeg_config_a"]
    _preset_cards(presets, selected_a_id, "eeg_config_a", "Choose a literature-informed configuration")
    preset_a = _preset_from_id(presets, st.session_state["eeg_config_a"])

    comparison_options = {preset.preset_id: preset.label for preset in presets}
    default_b_index = min(1, len(presets) - 1)
    selected_b_id = st.selectbox("Comparison configuration", list(comparison_options), index=default_b_index, format_func=lambda value: comparison_options[value], key="eeg_config_b")
    preset_b = _preset_from_id(presets, selected_b_id)

    st.markdown("<div class='section-heading compact'><h2>Simulation controls</h2><p>Sampling frequency is fixed at 256 Hz. The focus multiplier changes the selected configuration emphasis; noise, duration, seed and the additional multiplier affect only the generated signal and its derived displays.</p></div>", unsafe_allow_html=True)
    control_a, control_b, control_common = st.columns(3)
    with control_a:
        focus_gain_a = st.slider(f"{preset_a.focus_band} focus multiplier · A", .25, 2.5, 1.0, .05, key="eeg_focus_gain_a")
        noise_a = st.slider("Noise level · A (µV)", 0.0, 8.0, 2.0, .25, key="eeg_noise_a")
    with control_b:
        focus_gain_b = st.slider(f"{preset_b.focus_band} focus multiplier · B", .25, 2.5, 1.0, .05, key="eeg_focus_gain_b")
        noise_b = st.slider("Noise level · B (µV)", 0.0, 8.0, 2.0, .25, key="eeg_noise_b")
    with control_common:
        duration = st.slider("Simulation duration (s)", 6.0, 30.0, DEFAULT_DURATION_SECONDS, .5, key="eeg_duration")
        seed = st.number_input("Reproducibility seed", min_value=0, max_value=99999, value=42, step=1, key="eeg_seed")
        extra_band = st.selectbox("Additional illustrative band", list(BANDS), index=2, key="eeg_extra_band")
        extra_gain = st.slider("Additional band multiplier · B", .25, 2.5, 1.0, .05, key="eeg_extra_gain")
        st.button("Reset simulation controls", on_click=_reset_controls, key="eeg_reset", width="stretch")

    parameters_a = _build_parameters(preset_a, duration, int(seed), noise_a, focus_gain_a, preset_a.focus_band, 1.0)
    parameters_b = _build_parameters(preset_b, duration, int(seed), noise_b, focus_gain_b, extra_band, extra_gain)
    raw_a = generate_synthetic_raw(parameters_a)
    raw_b = generate_synthetic_raw(parameters_b)

    selection_left, selection_right = st.columns((1.4, .6))
    with selection_left:
        selected_channels = st.multiselect("Channels for signal and spectral analysis", list(CHANNELS), default=list(CHANNELS), key="eeg_channels")
    with selection_right:
        selected_electrode = st.selectbox("Highlight electrode", list(CHANNELS), key="eeg_selected_electrode")
    if not selected_channels:
        st.info("Select at least one channel to calculate the signal and spectrum.", icon="ℹ️")
        return
    window_end = min(float(duration), 6.0)
    time_window = st.slider("Displayed time interval (s)", 0.0, float(duration), (0.0, window_end), .5, key="eeg_window")
    selected_band = st.selectbox("Spatial view band", list(BANDS), index=list(BANDS).index(preset_a.focus_band), key="eeg_spatial_band")
    measure = st.radio("Spatial and comparison measure", ["relative", "absolute"], format_func=lambda value: "Relative power (%)" if value == "relative" else "Absolute power (V²)", horizontal=True, key="eeg_measure")

    analysis_a = analyse_raw(raw_a, selected_channels)
    analysis_b = analyse_raw(raw_b, selected_channels)
    tabs = st.tabs(["Signals", "Spectral analysis", "Spatial views", "Comparative laboratory", "Configuration provenance"])
    with tabs[0]:
        st.markdown("<div class='simulation-provenance'><strong>Configuration A:</strong> " + _safe(preset_a.label) + ". <strong>Configuration B:</strong> " + (_safe(preset_b.label) if (focus_gain_b == 1.0 and extra_gain == 1.0 and noise_b == noise_a) else "computational variant with user-adjusted parameters") + ". Both traces use the selected seed, duration and channel set.</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(time_series_figure(raw_a, selected_channels, time_window, f"Configuration A · {preset_a.label}"), width="stretch")
        with right:
            st.plotly_chart(time_series_figure(raw_b, selected_channels, time_window, f"Configuration B · {preset_b.label}"), width="stretch")
    with tabs[1]:
        st.plotly_chart(psd_comparison_figure(analysis_a, analysis_b, "Configuration A", "Configuration B"), width="stretch")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(band_power_bar(analysis_a, measure, "Configuration A band power", PALETTE["cyan"]), width="stretch")
        with right:
            st.plotly_chart(channel_band_heatmap(analysis_a, measure, "Configuration A channel-by-band distribution"), width="stretch")
    with tabs[2]:
        selected_name, selected_description = ELECTRODE_GUIDE[selected_electrode]
        st.markdown(
            "<div class='electrode-guide-panel'><div><span>Selected electrode</span>"
            f"<strong>{_safe(selected_electrode)} · {_safe(selected_name)}</strong><p>{_safe(selected_description)}</p></div>"
            "<div><span>Montage layout</span><strong>Eight scalp positions from the maintained 10–20 montage</strong>"
            "<p>Fp1/Fp2 frontopolar · F3/F4 frontal · C3/C4 central · O1/O2 occipital. Left/right labels are anatomical conventions.</p></div>"
            "<div><span>How to read this view</span><strong>Geometry is a spatial index, not a source estimate</strong>"
            "<p>The highlighted point links the 3D view to its channel trace. The topographies interpolate calculated band values only across the selected scalp channels.</p></div></div>",
            unsafe_allow_html=True,
        )
        if len(selected_channels) < 3:
            st.info("Select at least three channels to calculate a scalp topography.", icon="ℹ️")
        else:
            left, right = st.columns(2)
            with left:
                topography_a = topomap_figure(analysis_a, selected_band, measure, f"Configuration A · {selected_band}")
                st.pyplot(topography_a, clear_figure=True, width="stretch")
            with right:
                topography_b = topomap_figure(analysis_b, selected_band, measure, f"Configuration B · {selected_band}")
                st.pyplot(topography_b, clear_figure=True, width="stretch")
            st.caption("Scalp maps interpolate calculated values from the simulated eight-channel signal. They support comparison of the displayed channel pattern, not cortical source localization or high-resolution spatial precision.")
        st.plotly_chart(electrode_3d_figure(selected_electrode), width="stretch")
    with tabs[3]:
        difference = comparison_frame(analysis_a, analysis_b, measure)
        _comparison_summary(difference)
        st.plotly_chart(_difference_heatmap(difference, measure), width="stretch")
        st.caption("All differences in this panel are recalculated from the two simulated signals. They are not a numerical comparison against a published patient group.")
    with tabs[4]:
        st.markdown("<div class='section-heading compact'><h2>Configuration provenance</h2><p>Use this panel to separate the documented research orientation from the explicit model inputs and the outputs calculated in EEG Lab.</p></div>", unsafe_allow_html=True)
        _configuration_provenance(preset_a, studies, findings, set_page)
