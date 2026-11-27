"""Reproducible, MNE-based processing for OA-BRAIN EEG Lab simulations.

The generated signals are computational illustrations. They are never patient recordings
or reconstructions of a publication's participants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import mne
import numpy as np
import pandas as pd


CHANNELS: tuple[str, ...] = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2")
SFREQ = 256.0
DEFAULT_DURATION_SECONDS = 12.0
# MNE's maintained 10–20 coordinate set (the legacy ``standard_1020`` alias is deprecated).
MONTAGE_NAME = "colin27_1020"

# The requested analysis ranges are contiguous at shared boundaries. Integration
# treats every upper bound as exclusive except Gamma's final 45 Hz bound.
BANDS: dict[str, tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}
BAND_CENTERS_HZ = {"Delta": 2.0, "Theta": 6.0, "Alpha": 10.0, "Beta": 20.0, "Gamma": 36.0}

# Technical initialisation values in microvolts. They are user-adjustable model
# inputs, not published amplitudes and not an empirical OA profile.
ILLUSTRATIVE_COMPONENT_AMPLITUDES_UV = {
    "Delta": 5.0,
    "Theta": 4.0,
    "Alpha": 8.0,
    "Beta": 3.0,
    "Gamma": 1.0,
}


@dataclass(frozen=True)
class SimulationParameters:
    """User-controlled computational parameters with explicit units."""

    duration_seconds: float = DEFAULT_DURATION_SECONDS
    seed: int = 42
    noise_uv: float = 2.0
    focus_band: str = "Alpha"
    focus_channels: tuple[str, ...] = field(default_factory=tuple)
    focus_gain: float = 1.0
    band_scales: Mapping[str, float] = field(default_factory=dict)


def _valid_channels(channels: Sequence[str] | None) -> tuple[str, ...]:
    requested = CHANNELS if channels is None else tuple(channels)
    valid = tuple(channel for channel in requested if channel in CHANNELS)
    if not valid:
        raise ValueError("Select at least one valid EEG channel.")
    return valid


def _make_info() -> mne.Info:
    info = mne.create_info(list(CHANNELS), sfreq=SFREQ, ch_types="eeg")
    info.set_montage(MONTAGE_NAME, match_case=False, on_missing="raise")
    return info


def electrode_positions() -> dict[str, np.ndarray]:
    """Return 10–20 coordinates used by the MNE-compatible simulated Raw object."""
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    positions = montage.get_positions()["ch_pos"]
    return {channel: np.asarray(positions[channel], dtype=float) for channel in CHANNELS}


def generate_synthetic_raw(parameters: SimulationParameters) -> mne.io.BaseRaw:
    """Generate a reproducible eight-channel signal in volts and wrap it in MNE RawArray.

    A focus gain changes only a user-selected computational representation. It must not
    be interpreted as an effect magnitude reported by the linked publication.
    """
    if parameters.focus_band not in BANDS:
        raise ValueError(f"Unknown focus band: {parameters.focus_band}")
    if not 2.0 <= float(parameters.duration_seconds) <= 60.0:
        raise ValueError("Duration must be between 2 and 60 seconds.")
    if parameters.noise_uv < 0:
        raise ValueError("Noise must be zero or positive.")
    if parameters.focus_gain <= 0:
        raise ValueError("Focus gain must be greater than zero.")

    sample_count = int(round(float(parameters.duration_seconds) * SFREQ))
    times = np.arange(sample_count, dtype=float) / SFREQ
    rng = np.random.default_rng(int(parameters.seed))
    focus_channels = set(_valid_channels(parameters.focus_channels)) if parameters.focus_channels else set()
    signal_uv = np.zeros((len(CHANNELS), sample_count), dtype=float)

    for channel_index, channel in enumerate(CHANNELS):
        for band, frequency_hz in BAND_CENTERS_HZ.items():
            scale = float(parameters.band_scales.get(band, 1.0))
            if scale < 0:
                raise ValueError(f"Band scale for {band} must be zero or positive.")
            amplitude_uv = ILLUSTRATIVE_COMPONENT_AMPLITUDES_UV[band] * scale
            if band == parameters.focus_band and channel in focus_channels:
                amplitude_uv *= float(parameters.focus_gain)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal_uv[channel_index] += amplitude_uv * np.sin(2.0 * np.pi * frequency_hz * times + phase)
        if parameters.noise_uv:
            signal_uv[channel_index] += rng.normal(0.0, float(parameters.noise_uv), sample_count)

    raw = mne.io.RawArray(signal_uv * 1e-6, _make_info(), verbose=False)
    return raw


def compute_psd(raw: mne.io.BaseRaw, channels: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Compute Welch PSD in V²/Hz for selected channels using MNE."""
    selected = _valid_channels(channels)
    sample_count = raw.n_times
    n_fft = min(1024, sample_count)
    if n_fft < 64:
        raise ValueError("The signal is too short for the requested Welch PSD configuration.")
    spectrum = raw.compute_psd(
        method="welch",
        fmin=BANDS["Delta"][0],
        fmax=BANDS["Gamma"][1],
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        picks=list(selected),
        verbose=False,
    )
    values, frequencies = spectrum.get_data(return_freqs=True)
    return values, frequencies, selected


def _band_mask(frequencies: np.ndarray, band: str) -> np.ndarray:
    if band not in BANDS:
        raise ValueError(f"Unknown EEG band: {band}")
    low, high = BANDS[band]
    return (frequencies >= low) & (frequencies <= high if band == "Gamma" else frequencies < high)


def band_power_from_psd(psd_values: np.ndarray, frequencies: np.ndarray, channels: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integrate Welch PSD with frequency resolution and calculate row-wise relative power."""
    if psd_values.shape[0] != len(channels):
        raise ValueError("PSD channel dimension does not match the supplied channel list.")
    rows: list[dict[str, float | str]] = []
    for channel_index, channel in enumerate(channels):
        for band in BANDS:
            mask = _band_mask(frequencies, band)
            if int(mask.sum()) < 2:
                power = np.nan
            else:
                power = float(np.trapezoid(psd_values[channel_index, mask], frequencies[mask]))
            rows.append({"Channel": channel, "Band": band, "Absolute power (V²)": power})
    absolute = pd.DataFrame(rows)
    totals = absolute.groupby("Channel", as_index=False)["Absolute power (V²)"].sum(min_count=1).rename(columns={"Absolute power (V²)": "_Total"})
    relative = absolute.merge(totals, on="Channel", how="left")
    relative["Relative power (%)"] = np.where(
        relative["_Total"] > 0,
        relative["Absolute power (V²)"] / relative["_Total"] * 100.0,
        np.nan,
    )
    relative = relative.drop(columns="_Total")
    return absolute, relative


def analyse_raw(raw: mne.io.BaseRaw, channels: Sequence[str] | None = None) -> dict[str, object]:
    """Return PSD and band-power summaries generated from the same simulated Raw signal."""
    psd_values, frequencies, selected = compute_psd(raw, channels)
    absolute, relative = band_power_from_psd(psd_values, frequencies, selected)
    return {
        "channels": selected,
        "frequencies": frequencies,
        "psd": psd_values,
        "absolute": absolute,
        "relative": relative,
    }


def average_psd(analysis: Mapping[str, object]) -> np.ndarray:
    return np.asarray(analysis["psd"], dtype=float).mean(axis=0)


def band_values(analysis: Mapping[str, object], band: str, measure: str = "relative") -> pd.Series:
    """Return calculated channel values for a topographic or comparative view."""
    if measure not in {"absolute", "relative"}:
        raise ValueError("measure must be 'absolute' or 'relative'.")
    frame = analysis[measure]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Analysis does not contain a band-power dataframe.")
    column = "Absolute power (V²)" if measure == "absolute" else "Relative power (%)"
    selected = frame[frame["Band"] == band].set_index("Channel")[column]
    return selected


def comparison_frame(left: Mapping[str, object], right: Mapping[str, object], measure: str = "relative") -> pd.DataFrame:
    """Calculate signal-derived right-minus-left band-power differences on matching channels."""
    if measure not in {"absolute", "relative"}:
        raise ValueError("measure must be 'absolute' or 'relative'.")
    column = "Absolute power (V²)" if measure == "absolute" else "Relative power (%)"
    left_frame = left[measure]
    right_frame = right[measure]
    if not isinstance(left_frame, pd.DataFrame) or not isinstance(right_frame, pd.DataFrame):
        raise TypeError("Both analyses must include band-power dataframes.")
    merged = left_frame[["Channel", "Band", column]].merge(
        right_frame[["Channel", "Band", column]], on=["Channel", "Band"], suffixes=(" A", " B"), validate="one_to_one"
    )
    merged["Difference (B − A)"] = merged[f"{column} B"] - merged[f"{column} A"]
    return merged
