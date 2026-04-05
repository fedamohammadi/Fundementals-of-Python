"""Shared CSS injected into every page."""

import streamlit as st

# ── Palette ──────────────────────────────────────────────────────────────────
PURPLE   = "#7C3AED"
CYAN     = "#22D3EE"
AMBER    = "#F59E0B"
EMERALD  = "#10B981"
ROSE     = "#F43F5E"
BG       = "#0D0D1B"
CARD     = "#14142B"
BORDER   = "#2D2D5B"
TEXT     = "#E2E8F0"
MUTED    = "#94A3B8"

PLOTLY_TEMPLATE = "plotly_dark"


def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color + alpha float to an rgba() string.
    Plotly 6 no longer accepts 8-digit hex (e.g. #7C3AED55) in colorscales.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

CHART_COLORS = [PURPLE, CYAN, AMBER, EMERALD, ROSE,
                "#818CF8", "#34D399", "#FBBF24", "#FB7185", "#38BDF8"]


def inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── App background ── */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(160deg, #0D0D1B 0%, #111128 60%, #0A0A18 100%);
    }}
    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0A0A18 0%, #0F0F22 100%);
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p {{
        color: {TEXT} !important;
    }}

    /* ── Cards ── */
    .ml-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }}
    .ml-card:hover {{
        border-color: {PURPLE};
        transform: translateY(-2px);
    }}

    /* ── Hero gradient text ── */
    .hero-title {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, {PURPLE} 0%, {CYAN} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.3rem;
    }}
    .hero-sub {{
        font-size: 1.15rem;
        color: {MUTED};
        margin-bottom: 1.8rem;
    }}

    /* ── Section headers ── */
    .section-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TEXT};
        border-left: 4px solid {PURPLE};
        padding-left: 0.7rem;
        margin: 1.5rem 0 0.8rem;
    }}

    /* ── Metric badges ── */
    .badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }}
    .badge-purple {{ background: {PURPLE}22; color: {PURPLE}; border: 1px solid {PURPLE}55; }}
    .badge-cyan   {{ background: {CYAN}22;   color: {CYAN};   border: 1px solid {CYAN}55; }}
    .badge-amber  {{ background: {AMBER}22;  color: {AMBER};  border: 1px solid {AMBER}55; }}
    .badge-green  {{ background: {EMERALD}22; color: {EMERALD}; border: 1px solid {EMERALD}55; }}
    .badge-rose   {{ background: {ROSE}22;   color: {ROSE};   border: 1px solid {ROSE}55; }}

    /* ── Info box ── */
    .info-box {{
        background: {PURPLE}18;
        border: 1px solid {PURPLE}44;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        color: {TEXT};
        font-size: 0.92rem;
        margin: 0.8rem 0;
    }}

    /* ── Divider ── */
    .ml-divider {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 1.5rem 0;
    }}

    /* ── Streamlit widget overrides ── */
    div[data-testid="metric-container"] {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {MUTED};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {PURPLE} !important;
        border-bottom-color: {PURPLE} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def card(title: str, body: str, badge: str = "", badge_style: str = "purple") -> None:
    badge_html = (f'<span class="badge badge-{badge_style}">{badge}</span>'
                  if badge else "")
    st.markdown(f"""
    <div class="ml-card">
        <div style="font-size:1rem;font-weight:700;color:{TEXT};margin-bottom:0.4rem;">
            {badge_html} {title}
        </div>
        <div style="font-size:0.88rem;color:{MUTED};line-height:1.55;">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def info(text: str) -> None:
    st.markdown(f'<div class="info-box">ℹ️  {text}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="ml-divider">', unsafe_allow_html=True)
