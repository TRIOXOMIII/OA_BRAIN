"""Consistent Plotly presentation primitives for OA-BRAIN."""
from __future__ import annotations

import plotly.graph_objects as go

from config import PALETTE


def theme_figure(figure: go.Figure, height: int | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,26,47,.35)",
        font=dict(family="Inter, Manrope, Segoe UI, sans-serif", color=PALETTE["text"]),
        title=dict(font=dict(size=17, color=PALETTE["text"]), x=0.01, y=.98, pad=dict(b=30)),
        margin=dict(l=62, r=42, t=142, b=74),
        hoverlabel=dict(bgcolor="#071A2F", font_color=PALETTE["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0),
        height=height,
    )
    figure.update_xaxes(
        gridcolor="rgba(167,189,204,.13)", zerolinecolor="rgba(167,189,204,.2)",
        automargin=True, title_standoff=18, tickfont=dict(size=12, color=PALETTE["text"]), title_font=dict(color=PALETTE["text"]),
    )
    figure.update_yaxes(
        gridcolor="rgba(167,189,204,.13)", zerolinecolor="rgba(167,189,204,.2)",
        automargin=True, title_standoff=18, tickfont=dict(size=12, color=PALETTE["text"]), title_font=dict(color=PALETTE["text"]),
    )
    return figure


def empty_chart_message(message: str) -> None:
    import streamlit as st
    st.info(message, icon="ℹ️")
