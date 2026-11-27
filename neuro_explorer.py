"""Interactive Plotly and MNE spatial views for the integrated EEG Lab."""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import PALETTE
from eeg_processing import BANDS, CHANNELS, MONTAGE_NAME, SFREQ, average_psd, band_values, electrode_positions
from utils.visualization_utils import theme_figure


BAND_COLORS = {
    "Delta": "#67B7DC",
    "Theta": "#A78BFA",
    "Alpha": "#2DD4D9",
    "Beta": "#F4C95D",
    "Gamma": "#FB8B7A",
}


def time_series_figure(raw: mne.io.BaseRaw, channels: Sequence[str], time_window: tuple[float, float], title: str) -> go.Figure:
    """Create a vertically offset interactive trace plot in microvolts."""
    selected = tuple(channel for channel in channels if channel in raw.ch_names)
    if not selected:
        raise ValueError("Select at least one channel for the signal view.")
    start, end = time_window
    if end <= start:
        raise ValueError("The selected time interval must have positive duration.")
    start_index = max(0, int(round(start * raw.info["sfreq"])))
    end_index = min(raw.n_times, int(round(end * raw.info["sfreq"])))
    data_uv = raw.get_data(picks=list(selected), start=start_index, stop=end_index) * 1e6
    times = np.arange(start_index, end_index, dtype=float) / raw.info["sfreq"]
    typical_amplitude = float(np.percentile(np.abs(data_uv), 98)) if data_uv.size else 1.0
    offset = max(typical_amplitude * 2.8, 10.0)
    figure = go.Figure()
    tick_values: list[float] = []
    for index, channel in enumerate(selected):
        baseline = (len(selected) - 1 - index) * offset
        tick_values.append(baseline)
        figure.add_trace(go.Scatter(
            x=times,
            y=data_uv[index] + baseline,
            mode="lines",
            name=channel,
            line=dict(width=1.15),
            customdata=np.full(data_uv[index].shape, channel),
            hovertemplate="Channel: %{customdata}<br>Time: %{x:.3f} s<br>Display amplitude: %{y:.2f} µV<extra></extra>",
        ))
    figure.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis=dict(title="Amplitude (µV, vertically offset)", tickvals=tick_values, ticktext=list(selected), showgrid=False),
        hovermode="x unified",
        height=max(480, 110 * len(selected)),
    )
    return theme_figure(figure)


def psd_comparison_figure(left: Mapping[str, object], right: Mapping[str, object], left_name: str, right_name: str) -> go.Figure:
    """Overlay averages calculated from two simulated MNE Raw objects."""
    frequencies = np.asarray(left["frequencies"], dtype=float)
    left_psd = average_psd(left) * 1e12
    right_psd = average_psd(right) * 1e12
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frequencies, y=left_psd, mode="lines", name=left_name, line=dict(color=PALETTE["cyan"], width=2)))
    figure.add_trace(go.Scatter(x=frequencies, y=right_psd, mode="lines", name=right_name, line=dict(color=PALETTE["violet"], width=2, dash="dash")))
    for band, (low, high) in BANDS.items():
        figure.add_vrect(x0=low, x1=high, fillcolor=BAND_COLORS[band], opacity=.07, line_width=0, annotation_text=band, annotation_position="top left")
    figure.update_layout(title="Average Welch power spectral density", xaxis_title="Frequency (Hz)", yaxis_title="PSD (µV²/Hz)", height=540)
    return theme_figure(figure)


def channel_band_heatmap(analysis: Mapping[str, object], measure: str, title: str) -> go.Figure:
    """Show calculated band power by channel as an interactive heatmap."""
    frame = analysis[measure]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Analysis must include a band-power dataframe.")
    value_column = "Relative power (%)" if measure == "relative" else "Absolute power (V²)"
    matrix = frame.pivot(index="Channel", columns="Band", values=value_column).reindex(columns=list(BANDS))
    figure = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=[[0, "#0D2842"], [.5, "#2DD4D9"], [1, "#F4C95D"]],
        colorbar=dict(title=value_column),
        hovertemplate="Channel: %{y}<br>Band: %{x}<br>Calculated value: %{z:.4g}<extra></extra>",
    ))
    figure.update_layout(title=title, xaxis_title="Frequency band", yaxis_title="Channel", height=460)
    return theme_figure(figure)


def band_power_bar(analysis: Mapping[str, object], measure: str, title: str, color: str) -> go.Figure:
    """Plot the mean across selected simulated channels, retaining unit-specific labels."""
    frame = analysis[measure]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Analysis must include a band-power dataframe.")
    value_column = "Relative power (%)" if measure == "relative" else "Absolute power (V²)"
    summary = frame.groupby("Band", as_index=False)[value_column].mean().set_index("Band").reindex(list(BANDS)).reset_index()
    figure = go.Figure(go.Bar(
        x=summary["Band"], y=summary[value_column], marker_color=color,
        customdata=summary[["Band"]].to_numpy(), hovertemplate="Band: %{customdata[0]}<br>Mean across selected channels: %{y:.5g}<extra></extra>",
    ))
    figure.update_layout(title=title, xaxis_title="Frequency band", yaxis_title=value_column, height=460)
    return theme_figure(figure)


def _info_for_channels(channels: Sequence[str]) -> mne.Info:
    info = mne.create_info(list(channels), sfreq=SFREQ, ch_types="eeg")
    info.set_montage(MONTAGE_NAME, match_case=False, on_missing="raise")
    return info


def topomap_figure(analysis: Mapping[str, object], band: str, measure: str, title: str):
    """Create an MNE topomap from calculated synthetic band-power values.

    The sparse eight-channel layout is a scalp-level display, not source localization.
    """
    values = band_values(analysis, band, measure).dropna()
    if len(values) < 3:
        raise ValueError("At least three selected channels are required for a scalp topography.")
    info = _info_for_channels(list(values.index))
    figure, axis = plt.subplots(figsize=(7.2, 5.8), facecolor="#0D2842")
    image, _ = mne.viz.plot_topomap(
        values.to_numpy(),
        info,
        axes=axis,
        show=False,
        names=list(values.index),
        contours=0,
        sensors=True,
        extrapolate="local",
        cmap="viridis",
    )
    unit = "Relative power (%)" if measure == "relative" else "Absolute power (V²)"
    colorbar = figure.colorbar(image, ax=axis, shrink=.78, pad=.08)
    colorbar.set_label(unit, color="#F8FCFF")
    colorbar.ax.tick_params(colors="#F8FCFF")
    axis.set_title(title, color="#F8FCFF", pad=16, fontsize=13)
    axis.set_facecolor("#0D2842")
    for text in axis.texts:
        text.set_color("#F8FCFF")
    figure.tight_layout()
    return figure


def electrode_3d_figure(selected_channel: str | None = None) -> go.Figure:
    """Render a simplified, rotatable head sphere and actual standard-montage positions."""
    coordinates = electrode_positions()
    azimuth = np.linspace(0, 2 * np.pi, 36)
    polar = np.linspace(0, np.pi, 18)
    azimuth_grid, polar_grid = np.meshgrid(azimuth, polar)
    x = np.sin(polar_grid) * np.cos(azimuth_grid)
    y = np.sin(polar_grid) * np.sin(azimuth_grid)
    z = np.cos(polar_grid)
    position_matrix = np.vstack([coordinates[channel] for channel in CHANNELS])
    position_matrix = position_matrix / np.linalg.norm(position_matrix, axis=1).max() * 1.02
    colors = [PALETTE["coral"] if channel == selected_channel else PALETTE["cyan"] for channel in CHANNELS]
    sizes = [10 if channel == selected_channel else 6 for channel in CHANNELS]
    figure = go.Figure()
    figure.add_trace(go.Surface(x=x, y=y, z=z, opacity=.16, showscale=False, colorscale=[[0, "#34556D"], [1, "#17324A"]], hoverinfo="skip"))
    figure.add_trace(go.Scatter3d(
        x=position_matrix[:, 0], y=position_matrix[:, 1], z=position_matrix[:, 2],
        mode="markers+text", text=list(CHANNELS), textposition="top center",
        marker=dict(size=sizes, color=colors, line=dict(color="#F8FCFF", width=1)),
        hovertemplate="Electrode: %{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
    ))
    figure.update_layout(
        title="10–20 electrode configuration",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.45, y=1.45, z=.95)),
        ),
        height=560,
    )
    return theme_figure(figure)
