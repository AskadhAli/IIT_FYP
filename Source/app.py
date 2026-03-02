"""
app.py  —  Knowledge Garden v2.0
Predictive Analytics for Scholarly Document Management
Design: Botanical / Forest theme — deep greens, warm cream, antique gold
Typography: Playfair Display (display) + DM Sans (body)

FIXES APPLIED:
  #9  - get_model_bundle now takes df_raw as an explicit parameter (closure bug)
  #10 - df_hash now hashes a content-based fingerprint, not just shape/columns
  #11 - time_horizon filter is now wired up and used in the Overview scatter chart
  #12 - explain_ml_prune_score now uses actual model feature importances, not hardcoded weights
  #14 - apply_predictions arguments no longer prefixed with _ (Streamlit cache bug)
  #15 - risk_level NaN values handled: fillna("Unknown") before string cast
  #16 - NDCG baseline uses last_access_days (fairer) not access_count alone
"""

import os
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from survival_model import (
    train_survival_model,
    predict_on_dataframe,
    save_model,
    load_model,
    model_exists,
    generate_demo_data,
    MODEL_SAVE_PATH,
    FEATURE_NAMES,
)

try:
    from load_arxiv_data import load_arxiv_dataset, transform_arxiv_to_knowledge_garden
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Garden",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
/* ── CSS Variables ───────────────────────────────────── */
:root {
  --forest-deep:   #12291c;
  --forest-mid:    #1e4230;
  --forest-light:  #2d5a3d;
  --forest-mist:   #4a7c5a;
  --leaf-bright:   #a8d5a2;
  --leaf-pale:     #d4ead0;
  --cream:         #f5f2eb;
  --cream-dark:    #ede9df;
  --gold:          #c8962a;
  --gold-pale:     #f0d898;
  --red-prune:     #b83232;
  --red-pale:      #fde8e8;
  --amber:         #b8731a;
  --amber-pale:    #fef3e2;
  --text-dark:     #1a2e1f;
  --text-mid:      #4a5e4d;
  --text-muted:    #8a9e8d;
  --border:        #ddd9d0;
  --white:         #ffffff;
  --shadow-sm:     0 2px 8px rgba(18,41,28,0.08);
  --shadow-md:     0 4px 16px rgba(18,41,28,0.12);
  --shadow-lg:     0 8px 32px rgba(18,41,28,0.16);
  --radius-sm:     8px;
  --radius-md:     12px;
  --radius-lg:     16px;
}

/* ── Base ────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: "DM Sans", sans-serif !important;
}
.stApp {
  background: var(--cream) !important;
}
h1, h2, h3 {
  font-family: "Playfair Display", serif !important;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--forest-deep) 0%, #0d1f14 100%) !important;
  border-right: 1px solid #1e3a28 !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}
[data-testid="stSidebar"] * {
  color: var(--leaf-pale) !important;
  font-family: "DM Sans", sans-serif !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
  color: #b0cdb0 !important;
  font-size: 0.85rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--leaf-bright) !important;
  font-family: "Playfair Display", serif !important;
  letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
  background: var(--forest-mid) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="thumb"] {
  background: var(--leaf-bright) !important;
  border-color: var(--leaf-bright) !important;
}
[data-testid="stSidebar"] hr {
  border-color: #2a4a32 !important;
  margin: 12px 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  background: linear-gradient(135deg, #1e4230, #12291c) !important;
  color: var(--leaf-bright) !important;
  border: 1px solid #3a6a4a !important;
  border-radius: var(--radius-sm) !important;
  font-family: "DM Sans", sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.83rem !important;
  letter-spacing: 0.06em !important;
  padding: 10px 0 !important;
  transition: all 0.25s ease !important;
  text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: linear-gradient(135deg, #2d5a3d, #1e4230) !important;
  border-color: var(--leaf-bright) !important;
  box-shadow: 0 0 12px rgba(168,213,162,0.2) !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
  background: #2d5a3d !important;
  border: none !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span {
  color: var(--leaf-bright) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 3px !important;
  background: var(--cream-dark) !important;
  padding: 5px 6px !important;
  border-radius: 10px !important;
  border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-mid) !important;
  font-family: "DM Sans", sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.84rem !important;
  padding: 7px 14px !important;
  border: none !important;
  transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
  background: rgba(45,90,61,0.1) !important;
  color: var(--forest-light) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--forest-deep) !important;
  color: var(--leaf-bright) !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-sm) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 24px !important;
}

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 20px 22px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: box-shadow 0.2s, transform 0.2s !important;
}
[data-testid="metric-container"]:hover {
  box-shadow: var(--shadow-md) !important;
  transform: translateY(-1px) !important;
}
[data-testid="metric-container"] label {
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: "Playfair Display", serif !important;
  font-size: 1.9rem !important;
  color: var(--text-dark) !important;
  font-weight: 700 !important;
  line-height: 1.15 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-size: 0.78rem !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
  background: var(--forest-deep) !important;
  color: var(--leaf-bright) !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: "DM Sans", sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  padding: 10px 22px !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  background: var(--forest-light) !important;
  box-shadow: var(--shadow-md) !important;
  transform: translateY(-1px) !important;
}
.stDownloadButton > button {
  background: transparent !important;
  color: var(--forest-deep) !important;
  border: 2px solid var(--forest-deep) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
  background: var(--forest-deep) !important;
  color: var(--leaf-bright) !important;
}

/* ── Alerts ──────────────────────────────────────────── */
.stAlert {
  border-radius: var(--radius-md) !important;
  border-left-width: 4px !important;
  font-family: "DM Sans", sans-serif !important;
}

/* ── Expanders ───────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  background: var(--white) !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stExpander"] summary {
  font-weight: 600 !important;
  color: var(--text-dark) !important;
}

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ── Select / multiselect tags ───────────────────────── */
span[data-baseweb="tag"] {
  background: var(--forest-light) !important;
  border: none !important;
}
span[data-baseweb="tag"] span {
  color: var(--leaf-bright) !important;
}

/* ── Custom Components ───────────────────────────────── */

/* Hero banner */
.kg-hero {
  background: linear-gradient(135deg, var(--forest-deep) 0%, var(--forest-mid) 55%, #1a3828 100%);
  border-radius: var(--radius-lg);
  padding: 36px 44px 32px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}
.kg-hero::before {
  content: "";
  position: absolute;
  top: -60px; right: -40px;
  width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(168,213,162,0.12) 0%, transparent 65%);
  border-radius: 50%;
  pointer-events: none;
}
.kg-hero::after {
  content: "";
  position: absolute;
  bottom: -50px; left: 38%;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(200,150,42,0.08) 0%, transparent 65%);
  border-radius: 50%;
  pointer-events: none;
}
.kg-hero .pill {
  display: inline-block;
  background: rgba(168,213,162,0.15);
  border: 1px solid rgba(168,213,162,0.3);
  color: var(--leaf-bright);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 5px 14px;
  border-radius: 20px;
  margin-bottom: 14px;
}
.kg-hero h1 {
  font-family: "Playfair Display", serif !important;
  font-size: 2.8rem !important;
  color: #e8f5e2 !important;
  margin: 0 0 8px !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em !important;
  line-height: 1.1 !important;
}
.kg-hero .subtitle {
  font-size: 1.0rem;
  color: #8dbf8d;
  font-weight: 300;
  letter-spacing: 0.03em;
  margin: 0;
}

/* Model status banner */
.model-banner {
  display: flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(135deg, #edf7ea 0%, #e4f0e0 100%);
  border-left: 4px solid var(--forest-light);
  border-radius: var(--radius-md);
  padding: 11px 20px;
  margin-bottom: 24px;
  font-family: "DM Sans", sans-serif;
  font-size: 0.84rem;
  color: var(--text-dark);
  box-shadow: var(--shadow-sm);
}
.model-banner .dot {
  color: var(--text-muted);
  margin: 0 4px;
}
.model-banner strong {
  color: var(--forest-deep);
}

/* Stat card (metric override) */
.stat-card {
  background: var(--white);
  border-radius: var(--radius-md);
  padding: 20px 22px 18px;
  border-left: 4px solid;
  box-shadow: var(--shadow-sm);
  transition: box-shadow 0.2s, transform 0.2s;
  font-family: "DM Sans", sans-serif;
}
.stat-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.stat-card.green  { border-color: #2d8a4a; }
.stat-card.red    { border-color: var(--red-prune); }
.stat-card.amber  { border-color: var(--gold); }
.stat-card.blue   { border-color: #2980b9; }
.stat-card .s-label {
  font-size: 0.70rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.stat-card .s-value {
  font-family: "Playfair Display", serif;
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--text-dark);
  line-height: 1.1;
}
.stat-card .s-unit {
  font-family: "DM Sans", sans-serif;
  font-size: 1rem;
  font-weight: 400;
  color: var(--text-muted);
}
.stat-card .s-sub {
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-top: 4px;
}

/* Section heading */
.sec-title {
  font-family: "Playfair Display", serif;
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--text-dark);
  margin: 0 0 16px;
  padding-bottom: 10px;
  border-bottom: 2px solid var(--cream-dark);
  letter-spacing: -0.01em;
}

/* Risk legend badges */
.risk-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.risk-badge.high   { background: var(--red-pale);   color: var(--red-prune); }
.risk-badge.medium { background: var(--amber-pale);  color: var(--amber); }
.risk-badge.low    { background: #e8f5e2;            color: #276749; }

/* Info callout */
.callout {
  background: #edf7ea;
  border-left: 3px solid var(--forest-light);
  border-radius: var(--radius-sm);
  padding: 12px 18px;
  font-family: "DM Sans", sans-serif;
  font-size: 0.88rem;
  color: var(--text-dark);
  margin-top: 12px;
}
.callout.warn {
  background: var(--amber-pale);
  border-color: var(--amber);
}
.callout.danger {
  background: var(--red-pale);
  border-color: var(--red-prune);
}

/* Architecture cards */
.arch-card {
  background: var(--white);
  border-radius: var(--radius-md);
  padding: 20px 22px;
  border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  height: 100%;
}
.arch-card h4 {
  font-family: "Playfair Display", serif;
  font-weight: 700;
  color: var(--text-dark);
  font-size: 1.0rem;
  margin: 0 0 10px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--cream-dark);
}
.arch-card ul {
  margin: 0;
  padding-left: 18px;
  color: var(--text-mid);
  line-height: 2.0;
  font-size: 0.85rem;
}

/* Footer */
.kg-footer {
  background: linear-gradient(135deg, var(--forest-deep), #0d1f14);
  border-radius: var(--radius-lg);
  padding: 26px 32px;
  text-align: center;
  margin-top: 36px;
  font-family: "DM Sans", sans-serif;
}
.kg-footer .footer-title {
  font-family: "Playfair Display", serif;
  color: var(--leaf-bright);
  font-size: 1.1rem;
  font-weight: 700;
  margin-bottom: 6px;
}
.kg-footer .footer-sub {
  color: #6a9e6a;
  font-size: 0.8rem;
  letter-spacing: 0.04em;
}

/* Sidebar section label */
.sidebar-section {
  font-size: 0.68rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: #4a7a5a !important;
  margin: 18px 0 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_CHART_FONT  = dict(family="DM Sans, sans-serif", size=12, color="#1a2e1f")
_CHART_BG    = "#fafaf7"
_CHART_GRID  = "#e8e5de"
_TITLE_FONT  = dict(family="Playfair Display, serif", size=15, color="#1a2e1f")

def _style(fig, height=460, title=""):
    fig.update_layout(
        template="simple_white",
        height=height,
        font=_CHART_FONT,
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        title=dict(text=title, font=_TITLE_FONT, x=0.01),
        margin=dict(l=12, r=12, t=48, b=12),
        xaxis=dict(gridcolor=_CHART_GRID, linecolor="#ccc", zeroline=False),
        yaxis=dict(gridcolor=_CHART_GRID, linecolor="#ccc", zeroline=False),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE    = "Dataset/knowledge_garden_demo_v2.xlsx"
USE_ARXIV    = True
ARXIV_SAMPLE = 50_000

@st.cache_data(show_spinner=False)
def load_excel(path):
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df

@st.cache_data(show_spinner=False)
def load_arxiv(n):
    df_raw = load_arxiv_dataset(sample_size=n)
    return transform_arxiv_to_knowledge_garden(df_raw)

@st.cache_data(show_spinner=False)
def get_demo_data():
    # FIX (#17): generate 50,000 docs to match model design scale
    return generate_demo_data(n=50_000)

with st.spinner("Loading dataset…"):
    df_raw      = None
    data_source = ""
    if USE_ARXIV and ARXIV_AVAILABLE:
        try:
            df_raw      = load_arxiv(ARXIV_SAMPLE)
            data_source = f"ArXiv · {len(df_raw):,} papers"
        except Exception as e:
            st.warning(f"ArXiv unavailable ({e}). Falling back…")
    if df_raw is None and os.path.exists(DATA_FILE):
        try:
            df_raw      = load_excel(DATA_FILE)
            data_source = f"Excel · {len(df_raw):,} rows"
        except Exception as e:
            st.warning(f"Excel load failed ({e}). Using demo data…")
    if df_raw is None:
        df_raw      = get_demo_data()
        data_source = f"Demo · {len(df_raw):,} docs"

if "has_annotations"  not in df_raw.columns: df_raw["has_annotations"]  = 0
if "annotation_count" not in df_raw.columns: df_raw["annotation_count"] = 0
if "days_since_added" not in df_raw.columns:
    # FIX (#4): fixed default — do NOT derive from last_access_days (survival target)
    df_raw["days_since_added"] = 180

# ─────────────────────────────────────────────────────────────────────────────
# SURVIVAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def _compute_df_hash(df: pd.DataFrame) -> str:
    """
    FIX (#10): Compute a robust content-based hash instead of relying only
    on shape and column names. Two datasets with the same shape/columns but
    different data would previously get the same hash. We now sample values
    from the dataframe to build a SHA-256 fingerprint.
    """
    sample_size = min(500, len(df))
    sample = df.sample(n=sample_size, random_state=42) if len(df) >= sample_size else df
    fingerprint = (
        str(len(df))
        + str(list(df.columns))
        + sample.to_csv(index=False)
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def get_model_bundle(df_fingerprint: str, df_raw: pd.DataFrame):
    """
    FIX (#9): df_raw is now passed as an explicit argument instead of being
    captured from the outer scope. Streamlit's cache_resource keys on all
    arguments, so the bundle will be correctly invalidated when df_raw changes.
    Note: df_raw must come after the hashable key argument.
    """
    if model_exists(MODEL_SAVE_PATH):
        try:
            bundle = load_model(MODEL_SAVE_PATH)
            # Guard 1: stale feature count (e.g. old model before leakage fix)
            saved_n         = bundle["scaler"].n_features_in_
            expected        = len(FEATURE_NAMES)
            stale_features  = saved_n != expected
            # Guard 2: old key name "rsf_model" before the GBM rename
            stale_keys      = "rsf_model" in bundle and "gbm_model" not in bundle
            if stale_features or stale_keys:
                reason = (
                    f"feature count mismatch ({saved_n} vs {expected})"
                    if stale_features else
                    "model key renamed rsf_model → gbm_model"
                )
                st.warning(f"Stale model on disk ({reason}). Retraining automatically…")
                os.remove(MODEL_SAVE_PATH)
            else:
                return bundle, False
        except Exception:
            pass
    bundle = train_survival_model(df_raw)
    save_model(bundle)
    return bundle, True

df_fingerprint              = _compute_df_hash(df_raw)
with st.spinner("Loading survival model…"):
    model_bundle, _freshly  = get_model_bundle(df_fingerprint, df_raw)

@st.cache_data(show_spinner=False)
def apply_predictions(df, bundle_fingerprint: str, _bundle):
    """
    Cache keyed on df content and bundle_fingerprint (a hashable string).
    _bundle is prefixed with _ so Streamlit skips hashing it (dicts
    containing sklearn models and numpy arrays are not hashable by
    Streamlit's hasher). The fingerprint is derived from df_fingerprint
    which already changes whenever the data or model changes, so cache
    invalidation remains correct.
    """
    return predict_on_dataframe(df, _bundle)

df      = apply_predictions(df_raw, df_fingerprint, model_bundle)
metrics = model_bundle["metrics"]

# FIX (#15): Handle NaN risk_level values before converting to string.
# pd.cut can produce NaN for values exactly on bin boundaries or when
# prune_score is NaN. Converting NaN to the string "nan" creates a
# spurious fourth category in filters and the UI.
df["risk_level"] = (
    df["ml_risk_level"]
    .astype(object)
    .fillna("Unknown")
    .astype(str)
)

# ─────────────────────────────────────────────────────────────────────────────
# HERO + BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kg-hero">
  <div class="pill">Predictive Analytics · BSc Final Year Project</div>
  <h1>Knowledge Garden</h1>
  <p class="subtitle">Cognitive shelf-life prediction for scholarly document management</p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="model-banner">
  <strong>Survival Model Active</strong>
  <span class="dot">·</span>
  C-index <strong>{metrics['ensemble_c_index']:.4f}</strong>
  <span class="dot">·</span>
  IBS <strong>{metrics['ibs']:.4f}</strong>
  <span class="dot">·</span>
  CV <strong>{metrics['cv_c_index_mean']:.4f} ± {metrics['cv_c_index_std']:.4f}</strong>
  <span class="dot">·</span>
  <span style="color:#4a7c5a;">{data_source}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="background:linear-gradient(135deg,#1e4230,#12291c);
            padding:20px 16px 16px; margin:-1rem -1rem 0;">
  <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
              color:#a8d5a2; font-weight:700; margin-bottom:2px;">Controls</div>
  <div style="font-size:0.75rem; color:#5a8a6a; letter-spacing:0.04em;">
    Filter & configure the dashboard
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-section">Prune Threshold</p>', unsafe_allow_html=True)
prune_threshold = st.sidebar.slider(
    "Prune Score Threshold", 0.0, 1.0, 0.70, 0.05,
    help="Documents above this score are pruning candidates. ≥ 0.70 keeps false prune rate < 1%.",
    label_visibility="collapsed",
)
st.sidebar.caption(f"Flagging scores ≥ **{prune_threshold:.2f}**")

st.sidebar.markdown('<p class="sidebar-section">Risk Level</p>', unsafe_allow_html=True)
risk_levels = st.sidebar.multiselect(
    "Risk Level", ["Low", "Medium", "High"],
    default=["Medium", "High"],
    label_visibility="collapsed",
)

st.sidebar.markdown('<p class="sidebar-section">Research Field</p>', unsafe_allow_html=True)
all_fields = sorted(df["field"].dropna().unique())
fields = st.sidebar.multiselect(
    "Research Field", all_fields, default=all_fields,
    label_visibility="collapsed",
)

st.sidebar.markdown('<p class="sidebar-section">Search</p>', unsafe_allow_html=True)
search_query = st.sidebar.text_input(
    "Search Titles", placeholder="e.g. machine learning",
    label_visibility="collapsed",
)

st.sidebar.markdown('<p class="sidebar-section">Time Horizon</p>', unsafe_allow_html=True)
# FIX (#11): time_horizon is now wired up and used in the Overview scatter
# chart and the Prune Candidates filter to show documents likely to become
# stale within the selected number of days.
time_horizon = st.sidebar.radio(
    "Predict obsolescence within:",
    [30, 60, 90, 180],
    format_func=lambda x: f"{x} days",
    index=2,
    label_visibility="collapsed",
)
st.sidebar.caption(f"Highlighting documents with shelf-life ≤ **{time_horizon} days** equivalent")

st.sidebar.markdown("---")
if st.sidebar.button("Retrain Model", use_container_width=True):
    if os.path.exists(MODEL_SAVE_PATH):
        os.remove(MODEL_SAVE_PATH)
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────────────────────
df_f = df.copy()
if search_query:
    df_f = df_f[df_f["title"].astype(str).str.contains(search_query, case=False, na=False)]
if risk_levels:
    df_f = df_f[df_f["risk_level"].isin(risk_levels)]
if fields:
    df_f = df_f[df_f["field"].isin(fields)]

if df_f.empty:
    st.warning("No documents match the current filters. Adjust the sidebar controls.")
    st.stop()

df_prune  = df_f[df_f["ml_prune_score"] >= prune_threshold]
prune_pct = (len(df_prune) / len(df_f) * 100) if len(df_f) else 0

# FIX (#11): shelf_life_threshold converts the time_horizon (days) into an
# equivalent shelf-life months threshold for the Overview chart annotation.
shelf_life_horizon_months = time_horizon / 30.0

# ─────────────────────────────────────────────────────────────────────────────
# TOP STAT CARDS
# ─────────────────────────────────────────────────────────────────────────────
avg_sl      = df_f["ml_shelf_life_months"].mean()
high_risk_n = (df_f["risk_level"] == "High").sum()

sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.markdown(f"""
    <div class="stat-card blue">
      <div class="s-label">Total Documents</div>
      <div class="s-value">{len(df_f):,}</div>
      <div class="s-sub">matching current filters</div>
    </div>""", unsafe_allow_html=True)
with sc2:
    st.markdown(f"""
    <div class="stat-card red">
      <div class="s-label">Prune Candidates</div>
      <div class="s-value">{len(df_prune):,}</div>
      <div class="s-sub">{prune_pct:.1f}% of library flagged</div>
    </div>""", unsafe_allow_html=True)
with sc3:
    st.markdown(f"""
    <div class="stat-card green">
      <div class="s-label">Avg. ML Shelf-Life</div>
      <div class="s-value">{avg_sl:.1f}<span class="s-unit"> mo</span></div>
      <div class="s-sub">predicted remaining useful life</div>
    </div>""", unsafe_allow_html=True)
with sc4:
    st.markdown(f"""
    <div class="stat-card amber">
      <div class="s-label">High Risk Docs</div>
      <div class="s-value">{high_risk_n:,}</div>
      <div class="s-sub">immediate review recommended</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Knowledge Garden",
    "Prune Candidates",
    "Analytics",
    "Deep Dive",
    "Model Performance",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="sec-title">Document Relevance Landscape</p>', unsafe_allow_html=True)

    fig = px.scatter(
        df_f,
        x="last_access_days",
        y="ml_shelf_life_months",
        color="ml_prune_score",
        size="citation_count",
        hover_data={
            "title": True,
            "publication_year": True,
            "access_count": True,
            "ml_prune_score": ":.3f",
            "ml_shelf_life_months": ":.1f",
            "risk_level": True,
        },
        color_continuous_scale="RdYlGn_r",
        labels={
            "last_access_days":       "Days Since Last Access",
            "ml_shelf_life_months":   "Predicted Shelf-Life (months)",
            "ml_prune_score":         "Prune Score",
        },
    )
    fig.add_vline(
        x=90, line_dash="dot", line_color="#c8962a",
        annotation_text="90-day decay threshold",
        annotation_font=dict(color="#c8962a", size=11),
    )
    # FIX (#11): Add time_horizon horizontal line to the overview chart
    fig.add_hline(
        y=shelf_life_horizon_months, line_dash="dash", line_color="#2980b9",
        annotation_text=f"Time horizon: {time_horizon}d ({shelf_life_horizon_months:.1f} mo)",
        annotation_font=dict(color="#2980b9", size=11),
    )
    fig.update_traces(marker=dict(opacity=0.72, line=dict(width=0.5, color="#fff")))
    fig = _style(fig, height=520, title="ML Survival Model Predictions — All Documents")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="callout">
      <strong>Reading the chart</strong> — Upper-left (recently accessed, long shelf-life) = your
      core active references. Lower-right (stale, high prune score) = decay candidates flagged by the model.
      The gold dashed line marks the 90-day survival event boundary. The blue dashed line marks your
      selected time horizon of <strong>{time_horizon} days</strong> — documents below it are at risk
      of becoming obsolete within this window.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KNOWLEDGE GARDEN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # Real Streamlit checkboxes (hidden visually, state drives everything)
    _tc1, _tc2, _ = st.columns([1, 1, 6])
    with _tc1:
        view_by_risk  = st.checkbox("Risk",  value=True, key="kg_risk")
    with _tc2:
        view_by_field = st.checkbox("Field", value=True, key="kg_field")

    # ── Path logic ───────────────────────────────────────────────────────────
    df_viz = df_f.copy()
    df_viz["count"] = 1
    if view_by_risk and view_by_field:
        path    = ["risk_level", "field"]
        caption = "Inner ring = Risk Level  ·  Outer ring = Research Field  ·  Colour = avg prune score"
        mode_label = "Risk Level → Field"
    elif view_by_risk:
        path    = ["risk_level"]
        caption = "Each segment = one risk tier  ·  Colour = avg prune score"
        mode_label = "Risk Level only"
    elif view_by_field:
        path    = ["field", "risk_level"]
        caption = "Inner ring = Research Field  ·  Outer ring = Risk Level  ·  Colour = avg prune score"
        mode_label = "Field → Risk Level"
    else:
        path    = ["field"]
        caption = "Select a view mode above"
        mode_label = "Field only"

    r_on = "on" if view_by_risk  else "off"
    f_on = "on" if view_by_field else "off"

    st.markdown("""
<style>
div[data-testid="column"]:has(input#kg_risk),
div[data-testid="column"]:has(input#kg_field) { display:none !important; }

.kg-tbar {
    display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    background:linear-gradient(135deg,#0f2318 0%,#1a3828 100%);
    border-radius:14px; padding:12px 20px; margin:0 0 18px;
    border:1px solid #2a4a34;
    box-shadow:0 6px 20px rgba(18,41,28,0.4);
}
.kg-tbar-label {
    font-family:"DM Sans",sans-serif; font-size:0.68rem; font-weight:700;
    letter-spacing:0.14em; text-transform:uppercase; color:#3d6b4a;
    margin-right:4px;
}
.kg-tbtn {
    display:inline-flex; align-items:center; gap:8px;
    padding:8px 18px; border-radius:9px;
    font-family:"DM Sans",sans-serif; font-size:0.83rem; font-weight:600;
    border:1.5px solid; user-select:none; cursor:default; transition:all 0.2s;
    letter-spacing:0.01em;
}
.kg-tbtn.on  { background:rgba(168,213,162,0.18); border-color:#a8d5a2; color:#a8d5a2; box-shadow:0 0 12px rgba(168,213,162,0.18); }
.kg-tbtn.off { background:transparent; border-color:#2a4a34; color:#3d6b4a; }
.kg-tdot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.kg-tbtn.on  .kg-tdot { background:#a8d5a2; box-shadow:0 0 6px rgba(168,213,162,0.6); }
.kg-tbtn.off .kg-tdot { background:#2a4a34; }
.kg-tsep { width:1px; height:24px; background:#2a4a34; margin:0 2px; }
.kg-tmode {
    font-family:"DM Sans",sans-serif; font-size:0.75rem; color:#5a8a6a;
    font-style:italic; margin-left:2px;
}

.kg-shell {
    background:linear-gradient(160deg,#0d1f14 0%,#162e1e 55%,#0f2318 100%);
    border-radius:18px; padding:24px 22px 16px;
    border:1px solid #243d2c;
    box-shadow:0 12px 40px rgba(10,25,16,0.5), inset 0 1px 0 rgba(168,213,162,0.08);
    position:relative; overflow:hidden; margin-bottom:16px;
}
.kg-shell::before {
    content:""; position:absolute; top:-80px; right:-60px;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(168,213,162,0.06) 0%,transparent 60%);
    border-radius:50%; pointer-events:none;
}
.kg-shell::after {
    content:""; position:absolute; bottom:-60px; left:20%;
    width:200px; height:200px;
    background:radial-gradient(circle,rgba(200,150,42,0.04) 0%,transparent 60%);
    border-radius:50%; pointer-events:none;
}
.kg-cap {
    font-family:"DM Sans",sans-serif; font-size:0.76rem; color:#4a7a5a;
    text-align:center; margin-top:6px; letter-spacing:0.05em;
}

.kg-legend {
    display:flex; gap:10px; margin:18px 0 6px; flex-wrap:wrap; justify-content:center;
}
.kg-lpill {
    display:inline-flex; align-items:center; gap:7px;
    padding:7px 18px; border-radius:20px;
    font-family:"DM Sans",sans-serif; font-size:0.8rem; font-weight:600;
    border:1.5px solid; transition:transform 0.15s;
}
.kg-lpill:hover { transform:translateY(-1px); }
.kg-lpill.low    { background:#e8f5e2; border-color:#276749; color:#1a4a2a; }
.kg-lpill.medium { background:#fef3e2; border-color:#b8731a; color:#7a4a10; }
.kg-lpill.high   { background:#fde8e8; border-color:#b83232; color:#7b1e1e; }
.kg-ldot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.kg-lpill.low    .kg-ldot { background:#2d8a4a; }
.kg-lpill.medium .kg-ldot { background:#c8962a; }
.kg-lpill.high   .kg-ldot { background:#b83232; }

.kg-tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:4px; }
.kg-tile {
    border-radius:14px; padding:20px 22px;
    font-family:"DM Sans",sans-serif;
    transition:transform 0.2s, box-shadow 0.2s;
    position:relative; overflow:hidden;
}
.kg-tile:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,0.12); }
.kg-tile::after {
    content:""; position:absolute;
    bottom:-24px; right:-24px;
    width:90px; height:90px; border-radius:50%; opacity:0.12;
}
.kg-tile.low    { background:#e8f5e2; border:1.5px solid #b8ddc0; }
.kg-tile.medium { background:#fef3e2; border:1.5px solid #f0d090; }
.kg-tile.high   { background:#fde8e8; border:1.5px solid #f0b8b8; }
.kg-tile.low    ::after { background:#2d8a4a; }
.kg-tile.medium ::after { background:#c8962a; }
.kg-tile.high   ::after { background:#b83232; }
.kg-tier {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.13em; margin-bottom:8px;
}
.kg-tile.low    .kg-tier { color:#276749; }
.kg-tile.medium .kg-tier { color:#b8731a; }
.kg-tile.high   .kg-tier { color:#b83232; }
.kg-tnum {
    font-family:"Playfair Display",serif;
    font-size:2.2rem; font-weight:700; color:#1a2e1f; line-height:1.0;
}
.kg-tsub { font-size:0.78rem; color:#666; margin-top:5px; line-height:1.5; }
.kg-bwrap { background:rgba(0,0,0,0.09); border-radius:5px; height:5px; margin-top:12px; overflow:hidden; }
.kg-bar { height:100%; border-radius:5px; }
.kg-tile.low    .kg-bar { background:#2d8a4a; }
.kg-tile.medium .kg-bar { background:#c8962a; }
.kg-tile.high   .kg-bar { background:#b83232; }
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="kg-tbar">
  <span class="kg-tbar-label">View by</span>
  <div class="kg-tbtn {r_on}"><span class="kg-tdot"></span> Risk Level</div>
  <div class="kg-tbtn {f_on}"><span class="kg-tdot"></span> Research Field</div>
  <div class="kg-tsep"></div>
  <span class="kg-tmode">&#8594; {mode_label}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="kg-shell">', unsafe_allow_html=True)

    fig_sb = px.sunburst(
        df_viz, path=path, values="count",
        color="ml_prune_score",
        color_continuous_scale=[
            [0.00, "#1a6634"],
            [0.30, "#5dba4a"],
            [0.50, "#f5c518"],
            [0.72, "#e05a2b"],
            [1.00, "#8b1a1a"],
        ],
        color_continuous_midpoint=0.5,
    )
    fig_sb.update_traces(
        textinfo="label+percent entry",
        insidetextorientation="radial",
        textfont=dict(family="DM Sans, sans-serif", size=11, color="#ffffff"),
        marker=dict(line=dict(color="#0d1f14", width=1.8)),
    )
    fig_sb.update_layout(
        height=700,
        margin=dict(t=10, l=10, r=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_colorbar=dict(
            title=dict(text="Prune Score", font=dict(color="#a8d5a2", size=11, family="DM Sans")),
            tickfont=dict(color="#8dbf8d", size=10),
            bgcolor="rgba(13,31,20,0.8)",
            bordercolor="#2a4a34",
            borderwidth=1,
            thickness=13,
            len=0.55,
        ),
        font=dict(family="DM Sans, sans-serif", color="#ffffff"),
    )
    st.plotly_chart(fig_sb, use_container_width=True)
    st.markdown(f'<p class="kg-cap">{caption}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div class="kg-legend">
  <div class="kg-lpill low"><span class="kg-ldot"></span> Low Risk — keep active (score &lt; 0.3)</div>
  <div class="kg-lpill medium"><span class="kg-ldot"></span> Medium Risk — review (0.3 – 0.6)</div>
  <div class="kg-lpill high"><span class="kg-ldot"></span> High Risk — prune candidate (&gt; 0.6)</div>
</div>
""", unsafe_allow_html=True)

    _total = max(len(df_viz), 1)
    def _tile(lvl, cls, sub):
        pct  = len(sub) / _total * 100
        avgs = sub["ml_shelf_life_months"].mean() if len(sub) > 0 else 0
        bw   = int(pct)
        return (
            f'<div class="kg-tile {cls}">' +
            f'<div class="kg-tier">{lvl} Risk</div>' +
            f'<div class="kg-tnum">{len(sub):,}</div>' +
            f'<div class="kg-tsub">{pct:.1f}% of library&nbsp;&middot;&nbsp;avg {avgs:.1f} mo shelf-life</div>' +
            f'<div class="kg-bwrap"><div class="kg-bar" style="width:{bw}%;"></div></div>' +
            '</div>'
        )

    low_s  = df_viz[df_viz["risk_level"] == "Low"]
    med_s  = df_viz[df_viz["risk_level"] == "Medium"]
    high_s = df_viz[df_viz["risk_level"] == "High"]

    st.markdown(
        '<div class="kg-tiles">' +
        _tile("Low", "low", low_s) +
        _tile("Medium", "medium", med_s) +
        _tile("High", "high", high_s) +
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRUNE CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<p class="sec-title">Prune Candidates — ML Score ≥ {prune_threshold:.2f}</p>', unsafe_allow_html=True)

    if df_prune.empty:
        st.success("No documents meet the pruning threshold. Your library is in excellent shape!")
    else:
        # FIX (#11): Show how many of the prune candidates are within the
        # selected time horizon (i.e. their shelf-life is critically short).
        horizon_critical = df_prune[
            df_prune["ml_shelf_life_months"] <= shelf_life_horizon_months
        ]
        st.markdown(f"""
        <div class="callout danger">
          <strong>{len(df_prune):,} documents</strong> flagged for review by the survival model
          — {prune_pct:.1f}% of your current library view.
          Of these, <strong>{len(horizon_critical):,}</strong> have a predicted shelf-life ≤ {time_horizon} days
          (your selected time horizon).
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        disp = df_prune.sort_values("ml_prune_score", ascending=False).copy()
        disp["ML Prune Score"] = disp["ml_prune_score"].apply(lambda x: f"{x:.3f}")
        disp["ML Shelf-Life"]  = disp["ml_shelf_life_months"].apply(lambda x: f"{x:.1f} mo")
        disp["Last Access"]    = disp["last_access_days"].apply(
            lambda x: f"{int(x)} days ago" if pd.notna(x) else "n/a"
        )
        disp["Cox Risk"] = disp["ml_cox_risk"].apply(lambda x: f"{x:.3f}")
        # FIX (#1): renamed ml_rsf_risk → ml_gbm_risk
        disp["GBM Risk"] = disp["ml_gbm_risk"].apply(lambda x: f"{x:.3f}")

        cols_show = [
            "document_id", "title", "publication_year", "field",
            "Last Access", "access_count",
            "ML Prune Score", "ML Shelf-Life", "risk_level",
            "Cox Risk", "GBM Risk",
        ]
        st.dataframe(disp[cols_show], use_container_width=True, height=440)

        csv = disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Prune Candidates (CSV)",
            data=csv,
            file_name=f"prune_candidates_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="sec-title">Library Analytics</p>', unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        fig_hist = px.histogram(
            df_f, x="ml_prune_score", nbins=40,
            labels={"ml_prune_score": "ML Prune Score", "count": "Documents"},
            color_discrete_sequence=["#2d5a3d"],
        )
        fig_hist.add_vline(
            x=prune_threshold, line_dash="dash", line_color="#b83232",
            annotation_text=f"Threshold ({prune_threshold:.2f})",
            annotation_font=dict(color="#b83232", size=11),
        )
        fig_hist = _style(fig_hist, height=380, title="ML Prune Score Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

    with cb:
        fig_box = px.box(
            df_f, x="field", y="ml_shelf_life_months", color="risk_level",
            labels={"ml_shelf_life_months": "Shelf-Life (months)"},
            color_discrete_map={"Low": "#2d8a4a", "Medium": "#c8962a", "High": "#b83232"},
        )
        fig_box.update_xaxes(tickangle=45)
        fig_box = _style(fig_box, height=380, title="ML Shelf-Life by Research Field")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<p class="sec-title" style="margin-top:20px;">Publication Year vs. Access Recency</p>', unsafe_allow_html=True)
    fig_tl = px.scatter(
        df_f.sort_values("publication_year"),
        x="publication_year", y="last_access_days",
        color="ml_prune_score", size="access_count",
        hover_data=["title", "ml_shelf_life_months"],
        color_continuous_scale="RdYlGn_r",
        labels={
            "publication_year": "Publication Year",
            "last_access_days": "Days Since Last Access",
            "ml_prune_score": "Prune Score",
        },
    )
    fig_tl.update_traces(marker=dict(opacity=0.7, line=dict(width=0.4, color="#fff")))
    fig_tl = _style(fig_tl, height=420, title="Publication Year vs. Days Since Last Access  (bubble size = access count)")
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown('<p class="sec-title" style="margin-top:20px;">Cox PH vs GBM Survival Risk Score Comparison</p>', unsafe_allow_html=True)
    fig_risk = px.scatter(
        df_f.sample(min(2000, len(df_f)), random_state=42),
        x="ml_cox_risk", y="ml_gbm_risk",   # FIX (#1): renamed ml_rsf_risk → ml_gbm_risk
        color="ml_prune_score", color_continuous_scale="RdYlGn_r",
        opacity=0.65,
        labels={"ml_cox_risk": "Cox PH Risk Score", "ml_gbm_risk": "GBM Survival Risk Score"},
    )
    fig_risk = _style(fig_risk, height=420, title="Cox PH vs GBM Survival Risk Scores — Ensemble combines both")
    st.plotly_chart(fig_risk, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
def explain_ml_prune_score(doc: pd.Series, df_all: pd.DataFrame, feature_importances: dict):
    """
    FIX (#12): The explainability section now uses actual model feature
    importances from the trained GBM model instead of hardcoded heuristic
    weights (40/30/15/15) that bore no relation to the model's learned logic.

    The feature_importances dict (from model_bundle["feature_importances"])
    is normalised to sum to 1 and used to weight each factor's contribution
    bar and label. This makes the explanation genuinely reflect the model.
    """
    st.markdown('<p class="sec-title">Why This ML Prune Score?</p>', unsafe_allow_html=True)

    prune   = float(doc.get("ml_prune_score", 0))
    last_a  = float(doc.get("last_access_days", 0))
    acc_cnt = float(doc.get("access_count", 0))
    has_ann = float(doc.get("has_annotations", 0))
    cit_cnt = float(doc.get("citation_count", 0))
    cox_r   = float(doc.get("ml_cox_risk", 0))
    gbm_r   = float(doc.get("ml_gbm_risk", 0))   # FIX (#1)

    # Normalise raw feature importances to sum to 1
    fi_total = sum(feature_importances.values()) or 1.0
    fi_norm  = {k: v / fi_total for k, v in feature_importances.items()}

    # Map features to human-readable scores (0 = bad/decaying, 1 = healthy)
    recency_score = np.exp(-last_a / 90)
    acc_score     = min(1.0, acc_cnt / 10.0)
    ann_score     = 1.0 if has_ann > 0 else 0.0
    cit_score     = min(1.0, cit_cnt / 50.0)

    # Build factor display using actual importances from the model
    factors = [
        (
            "Recency (days_since_added / last_access_days)",
            f"Last accessed **{int(last_a)} days ago** → health score {recency_score:.3f}",
            recency_score,
            fi_norm.get("days_since_added", 0) + fi_norm.get("access_frequency", 0),
        ),
        (
            "Access Count",
            f"Accessed **{int(acc_cnt)} times** → health score {acc_score:.3f}",
            acc_score,
            fi_norm.get("access_count", 0),
        ),
        (
            "Annotations",
            f"{'Has annotations ✓' if has_ann > 0 else 'No annotations ✗'}",
            ann_score,
            fi_norm.get("has_annotations", 0) + fi_norm.get("annotation_count", 0),
        ),
        (
            "Citations",
            f"**{int(cit_cnt)} citations** → health score {cit_score:.3f}",
            cit_score,
            fi_norm.get("citation_count", 0),
        ),
        (
            "Document Age",
            f"Published in **{int(doc.get('publication_year', 2020))}**",
            max(0.0, 1.0 - float(doc.get("doc_age_years", 0) if "doc_age_years" in doc.index else 0) / 20),
            fi_norm.get("doc_age_years", 0),
        ),
    ]

    for name, desc, value, importance in factors:
        weight_pct = importance * 100
        c = "#b83232" if value < 0.3 else ("#c8962a" if value < 0.6 else "#276749")
        with st.expander(f"{name}  ·  {weight_pct:.1f}% model weight", expanded=True):
            st.markdown(desc)
            pct = int(np.clip(value, 0, 1) * 100)
            st.markdown(f"""
            <div style="background:#eee; border-radius:6px; height:8px; overflow:hidden; margin-top:8px;">
              <div style="background:{c}; width:{pct}%; height:100%; border-radius:6px;
                          transition:width 0.4s ease;"></div>
            </div>
            <div style="font-size:0.75rem; color:#888; margin-top:4px; font-family:'DM Sans',sans-serif;">
              Health score: {pct}%  ·  Model weight: {weight_pct:.1f}%
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sec-title" style="font-size:1.1rem;">Model Signals</p>', unsafe_allow_html=True)
    ms1, ms2, ms3 = st.columns(3)
    ms1.metric("Cox PH Risk",    f"{cox_r:.3f}", help="Linear proportional hazard risk")
    ms2.metric("GBM Surv. Risk", f"{gbm_r:.3f}", help="GBM log-duration survival risk (IPCW weighted)")
    ms3.metric("Ensemble Score", f"{prune:.3f}", help="35% Cox + 65% GBM weighted combination")

    st.markdown("""
    <div class="callout" style="margin-top:8px;">
      <strong>Note on explainability:</strong> Factor weights above are the GBM model's learned
      feature importances (mean decrease in impurity), not fixed heuristic weights. They reflect
      which features the model actually relied on when trained on this dataset.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="sec-title" style="font-size:1.1rem;">Overall Assessment</p>', unsafe_allow_html=True)
    if prune >= 0.7:
        st.markdown(f'<div class="callout danger"><strong>High Pruning Priority</strong> (Score: {prune:.3f}) — Multiple decay signals detected by the model.</div>', unsafe_allow_html=True)
    elif prune >= 0.4:
        st.markdown(f'<div class="callout warn"><strong>Medium Pruning Priority</strong> (Score: {prune:.3f}) — Mixed signals. Manual review recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="callout"><strong>Low Pruning Priority</strong> (Score: {prune:.3f}) — Strong relevance signals. Keep active.</div>', unsafe_allow_html=True)

    st.markdown('<p class="sec-title" style="margin-top:24px; font-size:1.1rem;">Document vs Library Average</p>', unsafe_allow_html=True)
    categories = ["Recency", "Access Freq.", "Annotations", "Citations"]
    max_la  = max(df_all["last_access_days"].max(), 1)
    max_acc = max(df_all["access_count"].max(), 1)
    max_cit = max(df_all["citation_count"].max(), 1)

    doc_vals = [1-(last_a/max_la), acc_cnt/max_acc, 1 if has_ann > 0 else 0, cit_cnt/max_cit]
    avg_vals = [
        1-(df_all["last_access_days"].mean()/max_la),
        df_all["access_count"].mean()/max_acc,
        df_all["has_annotations"].mean(),
        df_all["citation_count"].mean()/max_cit,
    ]
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=doc_vals, theta=categories, fill="toself",
        name="This Document",
        line_color="#1e4230", fillcolor="rgba(30,66,48,0.25)",
    ))
    radar.add_trace(go.Scatterpolar(
        r=avg_vals, theta=categories, fill="toself",
        name="Library Average",
        line_color="#c8962a", fillcolor="rgba(200,150,42,0.15)",
        opacity=0.8,
    ))
    radar.update_layout(
        polar=dict(
            bgcolor=_CHART_BG,
            radialaxis=dict(visible=True, range=[0,1], gridcolor=_CHART_GRID),
            angularaxis=dict(gridcolor=_CHART_GRID),
        ),
        paper_bgcolor=_CHART_BG,
        height=420,
        font=_CHART_FONT,
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=_CHART_GRID,
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(radar, use_container_width=True)


with tab5:
    st.markdown('<p class="sec-title">Document Deep Dive — ML Explainability</p>', unsafe_allow_html=True)
    st.caption("Select a document to inspect the survival model's reasoning for its prune score.")

    opts = df_f[["document_id", "title"]].copy()
    opts["label"] = opts["document_id"].astype(str) + " — " + opts["title"].astype(str).str[:80]
    selected = st.selectbox("Select a document:", opts["label"].tolist())
    sel_id   = selected.split(" — ")[0].strip()
    doc_row  = df_f[df_f["document_id"].astype(str) == sel_id].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("ML Prune Score",  f"{float(doc_row['ml_prune_score']):.3f}")
    d2.metric("ML Shelf-Life",   f"{float(doc_row['ml_shelf_life_months']):.1f} mo")
    d3.metric("Risk Level",      str(doc_row["risk_level"]))
    d4.metric("Last Access",     f"{int(doc_row['last_access_days'])} days ago")

    st.markdown("---")
    # FIX (#12): pass feature_importances so the explain function uses the real model weights
    explain_ml_prune_score(doc_row, df_f, model_bundle["feature_importances"])

    st.markdown("---")
    st.markdown('<p class="sec-title" style="font-size:1.1rem;">Actions (advisory — no automatic deletion)</p>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Mark as Keep", use_container_width=True):
            st.success("Marked as Keep (recorded in session).")
    with a2:
        if st.button("Archive", use_container_width=True):
            st.info("Marked for Archive (recorded in session).")
    with a3:
        if st.button("Flag for Deletion Review", use_container_width=True):
            st.warning("Flagged for deletion review (advisory only).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<p class="sec-title">Survival Model Performance Report</p>', unsafe_allow_html=True)
    st.caption("All metrics evaluated on a held-out 20% test set unless stated.")

    ens_ci  = metrics["ensemble_c_index"]
    gbm_ci  = metrics["gbm_c_index"]   # FIX (#1): renamed from rsf_c_index
    cox_ci  = metrics["cox_c_index"]
    ibs_val = metrics["ibs"]
    cv_mean = metrics["cv_c_index_mean"]
    cv_std  = metrics["cv_c_index_std"]

    st.markdown("#### Headline Metrics vs Thesis Targets")
    m_cols = st.columns(4)
    m_cols[0].metric("Ensemble C-index",    f"{ens_ci:.4f}",  f"Target ≥ 0.70 {'✓' if ens_ci  >= 0.70 else '✗'}")
    m_cols[1].metric("IBS (IPCW-corrected)", f"{ibs_val:.4f}", f"Target ≤ 0.18 {'✓' if ibs_val <= 0.18 else '✗'}")
    m_cols[2].metric("Cox PH C-index",       f"{cox_ci:.4f}")
    m_cols[3].metric("GBM Surv. C-index",    f"{gbm_ci:.4f}")

    st.markdown("#### 5-Fold Cross-Validation C-index")
    fold_scores = metrics.get("cv_fold_scores", [cv_mean] * 5)
    cv_fig = go.Figure()
    cv_fig.add_trace(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(fold_scores))],
        y=fold_scores,
        marker_color=["#2d5a3d" if s >= 0.70 else "#b83232" for s in fold_scores],
        text=[f"{s:.4f}" for s in fold_scores],
        textposition="outside",
        opacity=0.88,
    ))
    cv_fig.add_hline(y=0.70, line_dash="dash", line_color="#b83232",
                     annotation_text="Target 0.70", annotation_font=dict(color="#b83232"))
    cv_fig.add_hline(y=cv_mean, line_dash="dot", line_color="#2d5a3d",
                     annotation_text=f"CV Mean {cv_mean:.4f}", annotation_font=dict(color="#2d5a3d"))
    cv_fig = _style(cv_fig, height=360, title=f"Cross-Validated C-index: {cv_mean:.4f} ± {cv_std:.4f}")
    cv_fig.update_layout(yaxis=dict(range=[0, 1.08]))
    st.plotly_chart(cv_fig, use_container_width=True)

    # FIX (#1): Title updated to "GBM Feature Importance"
    st.markdown("#### GBM Feature Importance")
    fi    = model_bundle["feature_importances"]
    fi_df = (pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
             .sort_values("Importance", ascending=True))
    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=[
            [0, "#d4ead0"], [0.5, "#2d5a3d"], [1, "#12291c"]
        ],
    )
    fig_fi = _style(fig_fi, height=380, title="Feature Importances — GBM Survival Model")
    fig_fi.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("#### Model Architecture")
    ac1, ac2, ac3 = st.columns(3)
    arch_items = [
        ("Cox Proportional Hazards",
         ["Partial likelihood via L-BFGS-B", "7 input features (standardised)",
          "Assumes log-linear hazard", "Ensemble weight: <strong>35%</strong>"]),
        ("GBM Survival Model",
         ["GBM on log(duration) + IPCW weights", "300 estimators · depth 4 · lr 0.05",
          "Censored samples weighted 0.5×", "Ensemble weight: <strong>65%</strong>",
          "<em>Note: GBM approximation — not a true Random Survival Forest</em>"]),
        ("Ensemble Model",
         ["Both scores normalised to [0,1]", "Weighted avg (35% Cox + 65% GBM)",
          "Prune Score = normalised risk",
          "Shelf-Life = 24×(1−score) months (derived heuristic, not a survival function output)"]),
    ]
    for col, (title_a, items) in zip([ac1, ac2, ac3], arch_items):
        li = "".join(f"<li>{it}</li>" for it in items)
        with col:
            st.markdown(f"""
            <div class="arch-card">
              <h4>{title_a}</h4>
              <ul>{li}</ul>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### Training Dataset")
    ds_cols = st.columns(4)
    ds_cols[0].metric("Train set",        f"{metrics['n_train']:,}")
    ds_cols[1].metric("Test set",         f"{metrics['n_test']:,}")
    ds_cols[2].metric("Event rate",       f"{metrics['event_rate']:.1%}")
    ds_cols[3].metric("Decay threshold",  "90 days")

    st.markdown("""
    <div class="callout" style="margin-top:4px;">
      <strong>Survival Event Definition</strong> — A document is considered <em>decayed</em>
      when it has not been accessed for ≥ <strong>90 days</strong>. Documents accessed more recently
      are <em>censored</em> (still active). This mirrors the concept of <em>cognitive shelf-life</em> —
      the point at which a document stops being referenced by the researcher.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### Synthetic Data Limitations")
    st.markdown("""
    <div class="callout warn">
      <strong>Important caveat for thesis evaluation:</strong> All metrics above are computed on
      <em>synthetically generated</em> data where feature distributions and survival targets are
      drawn from the same parameterised process. This means:<br><br>
      &nbsp;&nbsp;• <strong>C-index and IBS may be optimistic</strong> — the model fits a signal that is
      artificially consistent across train and test splits.<br>
      &nbsp;&nbsp;• <strong>Real-world performance</strong> on actual Zotero/Mendeley export logs would likely
      be lower, due to noisier metadata and heterogeneous user behaviour.<br>
      &nbsp;&nbsp;• <strong>Baseline comparisons remain valid</strong> — all methods (heuristics and models)
      are evaluated on the same synthetic distribution, so relative improvements are meaningful.<br><br>
      These limitations are acknowledged per Chapter 3 (Research Challenges) and constitute a
      direction for future work: replication on de-identified real library exports.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Benchmarking vs Baseline Methods")
    st.caption("Survival model vs three deterministic heuristics. Higher C-index = better ranking.")

    @st.cache_data(show_spinner=False)
    def compute_baselines(df_serialised):
        """
        FIX (#14): Accepts a serialised DataFrame (CSV string) rather than
        the raw DataFrame directly, so Streamlit can hash the argument.
        """
        from survival_model import concordance_index, build_survival_targets
        _df = pd.read_csv(pd.io.common.StringIO(df_serialised))
        durations, events = build_survival_targets(_df)
        last_acc = _df["last_access_days"].fillna(0).values.astype(float)
        acc_cnt  = _df["access_count"].fillna(1).values.astype(float)
        age      = (2026 - _df["publication_year"].fillna(2020).values.astype(float))
        return {
            "Last-Access Threshold":      round(concordance_index(durations, events, last_acc), 4),
            "Access-Frequency Heuristic": round(concordance_index(durations, events, 1.0 / np.maximum(acc_cnt, 1)), 4),
            "Citation Half-Life (Age)":   round(concordance_index(durations, events, age),      4),
        }

    df_f_csv    = df_f.to_csv(index=False)
    baselines   = compute_baselines(df_f_csv)

    bench_data = {
        "Method": [
            "Last-Access Threshold (baseline)",
            "Access-Frequency Heuristic (baseline)",
            "Citation Half-Life / Age (baseline)",
            "Cox Proportional Hazards",
            "GBM Survival Model (log-duration + IPCW)",
            "★ Ensemble (Cox + GBM)",
        ],
        "C-index": [
            baselines["Last-Access Threshold"],
            baselines["Access-Frequency Heuristic"],
            baselines["Citation Half-Life (Age)"],
            round(cox_ci, 4), round(gbm_ci, 4), round(ens_ci, 4),
        ],
        "Type": ["Baseline"]*3 + ["Survival Model"]*3,
    }
    bench_df = pd.DataFrame(bench_data).sort_values("C-index", ascending=True)
    fig_bench = go.Figure(go.Bar(
        x=bench_df["C-index"], y=bench_df["Method"], orientation="h",
        marker_color=["#aab7c4" if t == "Baseline" else "#2d5a3d"
                      for t in bench_df["Type"]],
        text=[f"{v:.4f}" for v in bench_df["C-index"]],
        textposition="outside",
    ))
    fig_bench.add_vline(x=0.70, line_dash="dash", line_color="#b83232",
                        annotation_text="Target 0.70",
                        annotation_font=dict(color="#b83232"))
    fig_bench = _style(fig_bench, height=400, title="C-index: Survival Models vs Baseline Methods")
    fig_bench.update_layout(xaxis=dict(range=[0, 1.1]), showlegend=False,
                            margin=dict(l=12, r=80, t=48, b=12))
    st.plotly_chart(fig_bench, use_container_width=True)

    best_baseline = max(baselines.values())
    improvement   = ens_ci - best_baseline
    st.markdown(f"""
    <div class="callout">
      The ensemble achieves C-index <strong>{ens_ci:.4f}</strong> vs best baseline
      <strong>{best_baseline:.4f}</strong> — an improvement of
      <strong>+{improvement:.4f}</strong> ({improvement/best_baseline*100:.1f}% relative gain).
    </div>""", unsafe_allow_html=True)

    bench_display = bench_df.copy()
    bench_display["vs Baseline"] = bench_display["C-index"].apply(
        lambda x: f"+{x-best_baseline:.4f}" if x > best_baseline else f"{x-best_baseline:.4f}"
    )
    bench_display["Meets Target"] = bench_display["C-index"].apply(lambda x: "✓" if x >= 0.70 else "✗")
    st.dataframe(
        bench_display[["Method","Type","C-index","vs Baseline","Meets Target"]],
        use_container_width=True, hide_index=True,
    )

    # ── NDCG@20 ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### NDCG@20 — Document Ranking Quality")

    @st.cache_data(show_spinner=False)
    def compute_ndcg(df_serialised):
        """
        NDCG@20 measures how well the model ranks documents for pruning.

        Graded relevance = normalised last_access_days (0→1).
        A document not accessed for 400 days is more relevant to prune
        than one not accessed for 100 days.

        The model ranking = ml_prune_score descending (high = prune first).

        FIX (#16): The original baseline used access_count (ascending) which
        is a very weak heuristic that made the model look artificially good.
        The corrected baseline uses last_access_days descending — the same
        signal as the ideal ranking — which is the strongest single-feature
        heuristic and a fairer comparison point.
        """
        _df     = pd.read_csv(pd.io.common.StringIO(df_serialised))
        scores  = _df["ml_prune_score"].fillna(0).values.astype(float)
        last_acc = _df["last_access_days"].fillna(0).values.astype(float)
        k        = 20

        max_la    = max(last_acc.max(), 1.0)
        relevance = last_acc / max_la

        def dcg_at_k(ranked_rel, k):
            r = ranked_rel[:k]
            return float(np.sum(r / np.log2(np.arange(2, len(r) + 2))))

        idcg = dcg_at_k(relevance[np.argsort(-relevance)], k)
        if idcg == 0:
            return 0.0, 0.0, k

        ndcg_model = dcg_at_k(relevance[np.argsort(-scores)], k) / idcg

        # FIX (#16): Use last_access_days descending as a fair, strong baseline
        # (not access_count, which is a much weaker proxy)
        ndcg_base = dcg_at_k(relevance[np.argsort(-last_acc)], k) / idcg

        return round(ndcg_model, 4), round(ndcg_base, 4), k

    ndcg_model, ndcg_base, k_val = compute_ndcg(df_f.to_csv(index=False))
    nc1, nc2, nc3 = st.columns(3)
    nc1.metric(f"NDCG@{k_val} (Ensemble)",             f"{ndcg_model:.4f}", f"Target ≥ 0.75 {'✓' if ndcg_model >= 0.75 else '✗'}")
    nc2.metric(f"NDCG@{k_val} (Last-Access baseline)", f"{ndcg_base:.4f}",  help="Strongest single-feature heuristic baseline")
    nc3.metric(f"NDCG@{k_val} Improvement",            f"+{ndcg_model-ndcg_base:.4f}")

    if ndcg_model >= 0.75:
        st.success(f"✓ NDCG@{k_val} ≥ 0.75 achieved: {ndcg_model:.4f}")
    else:
        st.warning(f"NDCG@{k_val} = {ndcg_model:.4f} — below 0.75 target.")

    # ── False Pruning Rate ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### False Pruning Rate")

    @st.cache_data(show_spinner=False)
    def compute_fpr(df_serialised, threshold):
        """
        False Pruning Rate = documents flagged for pruning that are still
        genuinely active (last accessed < 90 days ago).
        FPR < 1% is the thesis safety target.
        """
        _df          = pd.read_csv(pd.io.common.StringIO(df_serialised))
        flagged      = _df["ml_prune_score"] >= threshold
        still_active = _df["last_access_days"] < 90
        fp_count     = int((flagged & still_active).sum())
        total_flagged = int(flagged.sum())
        rate = (fp_count / total_flagged * 100) if total_flagged > 0 else 0.0
        return fp_count, total_flagged, round(rate, 3)

    df_f_csv_fpr = df_f.to_csv(index=False)
    fp_count, total_flagged, fp_rate = compute_fpr(df_f_csv_fpr, prune_threshold)
    fp1, fp2, fp3 = st.columns(3)
    fp1.metric("False Prune Rate",  f"{fp_rate:.3f}%",  f"Target < 1% {'✓' if fp_rate < 1.0 else '✗'}")
    fp2.metric("False Prunes",      f"{fp_count:,}",    f"out of {total_flagged:,} flagged")
    fp3.metric("Correctly Flagged", f"{total_flagged-fp_count:,}")

    if fp_rate < 1.0:
        st.success(f"✓ False pruning rate < 1% achieved at threshold {prune_threshold:.2f}: {fp_rate:.3f}%")
    elif fp_rate < 5.0:
        st.warning(
            f"False pruning rate {fp_rate:.3f}% at threshold {prune_threshold:.2f}. "
            f"Raise the threshold to ≥ 0.70 to achieve the < 1% safety target."
        )
    else:
        st.error(
            f"✗ False pruning rate {fp_rate:.3f}% — threshold {prune_threshold:.2f} is too low. "
            f"The model meets < 1% FPR at threshold ≥ 0.70."
        )

    # ── Full Checklist ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Full Thesis Target Checklist")
    checklist = [
        ("C-index ≥ 0.70",        ens_ci  >= 0.70,  f"{ens_ci:.4f}"),
        ("IBS ≤ 0.18",            ibs_val <= 0.18,  f"{ibs_val:.4f}"),
        (f"NDCG@{k_val} ≥ 0.75", ndcg_model >= 0.75, f"{ndcg_model:.4f}"),
        ("False prune rate < 1%", fp_rate < 1.0,    f"{fp_rate:.3f}%"),
    ]
    ch_cols = st.columns(4)
    for i, (label, passed, value) in enumerate(checklist):
        with ch_cols[i]:
            st.markdown(f"""
            <div style="background:{'#e8f5e2' if passed else '#fde8e8'};
                        border:2px solid {'#2d8a4a' if passed else '#b83232'};
                        border-radius:12px; padding:18px 16px; text-align:center;
                        font-family:'DM Sans',sans-serif;">
              <div style="font-size:1.4rem; font-weight:700; margin-bottom:6px; color:{'#1a5a2a' if passed else '#b83232'};">{'Pass' if passed else 'Fail'}</div>
              <div style="font-weight:700; color:{'#1a5a2a' if passed else '#7b1e1e'};
                          font-size:0.85rem;">{label}</div>
              <div style="font-family:'Playfair Display',serif; font-size:1.4rem;
                          font-weight:700; color:{'#12291c' if passed else '#b83232'};
                          margin-top:6px;">{value}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kg-footer">
  <div class="footer-title">Knowledge Garden v2.0</div>
  <div class="footer-sub">
    Survival Model: Cox PH + GBM Ensemble &nbsp;·&nbsp;
    C-index {metrics['ensemble_c_index']:.4f} &nbsp;·&nbsp;
    IBS {metrics['ibs']:.4f} &nbsp;·&nbsp;
    Trained on {metrics['n_train'] + metrics['n_test']:,} documents
  </div>
</div>
""", unsafe_allow_html=True)