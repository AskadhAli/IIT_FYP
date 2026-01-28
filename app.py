import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from load_arxiv_data import load_arxiv_dataset, transform_arxiv_to_knowledge_garden

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Knowledge Garden - Predictive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# STYLE
# ----------------------------
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .prune-high { color: #d62728; font-weight: bold; }
    .prune-medium { color: #ff7f0e; font-weight: bold; }
    .prune-low { color: #2ca02c; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# TITLE
# ----------------------------
st.title("🌱 Knowledge Garden")
st.markdown("### Predictive Analytics for Scholarly Document Management")
st.caption(
    "Interim prototype - Estimates cognitive shelf-life of documents using survival analysis (mock predictions)"
)

# ----------------------------
# DATA LOADING
# ----------------------------
USE_ARXIV_DATASET = True  # Set to False to use the original Excel file
DATA_FILE = "knowledge_garden_demo_v2.xlsx"
ARXIV_SAMPLE_SIZE = 10000  # Number of papers to load (None for all, but be careful - it's huge!)

@st.cache_data
def load_data_excel(path: str) -> pd.DataFrame:
    """Load data from Excel file (original method)"""
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip().str.lower()
    return df

@st.cache_data
def load_data_arxiv(sample_size: int = None) -> pd.DataFrame:
    """Load and transform ArXiv dataset"""
    df_arxiv = load_arxiv_dataset(sample_size=sample_size)
    df = transform_arxiv_to_knowledge_garden(df_arxiv)
    return df

# Load data based on configuration
if USE_ARXIV_DATASET:
    try:
        with st.spinner("Loading ArXiv dataset and transforming to Knowledge Garden format... This may take a minute."):
            df = load_data_arxiv(sample_size=ARXIV_SAMPLE_SIZE)
        st.success(f"Loaded {len(df):,} papers from ArXiv dataset")
    except Exception as e:
        st.error(f"Could not load ArXiv dataset: {e}")
        st.info("Falling back to Excel file...")
        try:
            df = load_data_excel(DATA_FILE)
        except Exception as e2:
            st.error(f"Could not load {DATA_FILE}: {e2}")
            st.stop()
else:
    try:
        df = load_data_excel(DATA_FILE)
    except Exception as e:
        st.error(f"Could not load {DATA_FILE}: {e}")
        st.stop()

# ----------------------------
# SCHEMA CHECK
# ----------------------------
required_cols = {
    "document_id",
    "title",
    "publication_year",
    "field",
    "last_access_days",
    "access_count",
    "citation_count",
    "prune_score",
    "predicted_shelf_life_months",
    "risk_level",
}

missing = required_cols - set(df.columns)
if missing:
    st.error(f"Dataset is missing required columns: {sorted(missing)}")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Optional columns used in Deep Dive (safe defaults if missing)
if "has_annotations" not in df.columns:
    df["has_annotations"] = 0
if "annotation_count" not in df.columns:
    df["annotation_count"] = 0

# Coerce numeric columns (prevents Plotly weirdness)
for col in [
    "prune_score",
    "predicted_shelf_life_months",
    "last_access_days",
    "access_count",
    "citation_count",
    "has_annotations",
    "annotation_count",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("⚙️ Filter Controls")

prune_threshold = st.sidebar.slider(
    "Prune Score Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Documents above this threshold are pruning candidates",
)

risk_levels = st.sidebar.multiselect(
    "Risk Level",
    options=["Low", "Medium", "High"],
    default=["Medium", "High"],
    help="Filter by document risk level",
)

fields = st.sidebar.multiselect(
    "Research Field",
    options=sorted(df["field"].dropna().unique()),
    default=sorted(df["field"].dropna().unique()),
    help="Filter by research domain",
)

search_query = st.sidebar.text_input("🔍 Search Titles", placeholder="e.g., machine learning")

st.sidebar.markdown("---")
st.sidebar.markdown("**Time Horizon**")
time_horizon = st.sidebar.radio(
    "Predict obsolescence within:",
    options=[30, 60, 90, 180],
    format_func=lambda x: f"{x} days",
    index=2,
)
st.sidebar.caption("Note: time horizon is a placeholder for final survival model risk estimation.")

# ----------------------------
# APPLY FILTERS
# ----------------------------
df_filtered = df.copy()

if search_query:
    df_filtered = df_filtered[
        df_filtered["title"].astype(str).str.contains(search_query, case=False, na=False)
    ]

if risk_levels:
    df_filtered = df_filtered[df_filtered["risk_level"].isin(risk_levels)]

if fields:
    df_filtered = df_filtered[df_filtered["field"].isin(fields)]

if df_filtered.empty:
    st.warning("No documents match the current filters. Adjust the sidebar controls.")
    st.stop()

df_prune_candidates = df_filtered[df_filtered["prune_score"] >= prune_threshold]

# ----------------------------
# METRICS
# ----------------------------
st.header("📊 Library Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Documents", f"{len(df_filtered):,}", help="Documents matching current filters")

with col2:
    prune_count = len(df_prune_candidates)
    prune_pct = (prune_count / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    st.metric("Prune Candidates", f"{prune_count:,}", f"{prune_pct:.1f}% of library", delta_color="inverse")

with col3:
    avg_shelf_life = df_filtered["predicted_shelf_life_months"].mean()
    st.metric("Avg. Shelf-Life", f"{avg_shelf_life:.1f} months", help="Average predicted remaining usefulness")

with col4:
    high_risk_count = (df_filtered["risk_level"] == "High").sum()
    st.metric("High Risk Docs", f"{high_risk_count:,}", help="Documents with high obsolescence risk")

st.markdown("---")

# ============================================================
# DEEP DIVE / EXPLAINABILITY FUNCTIONS
# ============================================================
def explain_prune_score(doc_row: pd.Series, df_all: pd.DataFrame):
    st.markdown("### 🔍 Why This Prune Score?")

    prune_score = float(doc_row.get("prune_score", 0))
    last_access = float(doc_row.get("last_access_days", 0))
    access_count = float(doc_row.get("access_count", 0))
    has_annotations = float(doc_row.get("has_annotations", 0))
    citation_count = float(doc_row.get("citation_count", 0))
    pub_year = int(doc_row.get("publication_year", 0)) if pd.notna(doc_row.get("publication_year", None)) else 0

    # Percentiles for comparison (safe)
    if df_all["last_access_days"].notna().any():
        last_access_pct = (df_all["last_access_days"] <= last_access).mean() * 100
    else:
        last_access_pct = 50

    if df_all["access_count"].notna().any():
        access_count_pct = (df_all["access_count"] >= access_count).mean() * 100
    else:
        access_count_pct = 50

    factors = []

    # Factor: recency
    if last_access > 180:
        factors.append(("⏰ Stale Access Pattern", f"Not accessed for **{int(last_access)} days** (staler than ~{100-last_access_pct:.0f}% of library)", 0.40))
    elif last_access > 90:
        factors.append(("⏰ Moderate Staleness", f"Last accessed **{int(last_access)} days** ago", 0.25))
    else:
        factors.append(("✅ Recently Accessed", f"Accessed **{int(last_access)} days** ago", 0.10))

    # Factor: engagement
    if access_count <= 2:
        factors.append(("📉 Low Engagement", f"Only accessed **{int(access_count)} times** (lower than most docs)", 0.30))
    elif access_count <= 5:
        factors.append(("📊 Moderate Usage", f"Accessed **{int(access_count)} times**", 0.20))
    else:
        factors.append(("📈 High Engagement", f"Accessed **{int(access_count)} times** (often used)", 0.10))

    # Factor: annotations
    if has_annotations <= 0:
        factors.append(("📝 No Annotations", "No highlights/notes found (weak personal value signal)", 0.15))
    else:
        factors.append(("✍️ Has Annotations", "Has highlights/notes (strong personal value signal)", 0.08))

    # Factor: age
    if pub_year > 0:
        doc_age_years = 2026 - pub_year
        if doc_age_years >= 8:
            factors.append(("📅 Aging Document", f"Published **{doc_age_years} years ago** ({pub_year})", 0.10))

    st.markdown("**Contributing Factors:**")
    for (name, desc, weight) in factors:
        with st.expander(f"{name} (weight ~{int(weight*100)}%)", expanded=True):
            st.markdown(desc)
            st.progress(min(max(weight, 0.0), 1.0))

    st.markdown("---")
    st.markdown("### 📊 Overall Assessment")

    if prune_score >= 0.7:
        st.error(
            f"**High Pruning Priority** (Score: {prune_score:.2f})\n\n"
            "This document shows multiple decline indicators. Review if it’s still needed."
        )
    elif prune_score >= 0.4:
        st.warning(
            f"**Medium Pruning Priority** (Score: {prune_score:.2f})\n\n"
            "Mixed signals. Keep if it supports current research, otherwise archive."
        )
    else:
        st.success(
            f"**Low Pruning Priority** (Score: {prune_score:.2f})\n\n"
            "Strong signals of ongoing relevance. Keep active."
        )

    st.markdown("### 📈 How This Document Compares")

    categories = ["Recency", "Access Freq.", "Annotations", "Citations"]

    # Normalize to 0-1 scale safely
    max_last = max(df_all["last_access_days"].max(), 1)
    max_access = max(df_all["access_count"].max(), 1)
    max_cite = max(df_all["citation_count"].max(), 1)

    doc_values = [
        1 - (last_access / max_last),
        access_count / max_access,
        1 if has_annotations > 0 else 0,
        citation_count / max_cite,
    ]

    avg_values = [
        1 - (df_all["last_access_days"].mean() / max_last),
        df_all["access_count"].mean() / max_access,
        df_all["has_annotations"].mean(),
        df_all["citation_count"].mean() / max_cite,
    ]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=doc_values, theta=categories, fill="toself", name="This Document"))
    radar.add_trace(go.Scatterpolar(r=avg_values, theta=categories, fill="toself", name="Library Average", opacity=0.6))

    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(radar, use_container_width=True)

def deep_dive_tab(df_view: pd.DataFrame):
    st.subheader("🔍 Document Deep Dive")
    st.caption("Select a document and see why it was flagged (interpretability view).")

    # Select by document_id to avoid duplicate titles
    options = df_view[["document_id", "title"]].copy()
    options["label"] = options["document_id"].astype(str) + " — " + options["title"].astype(str)

    selected_label = st.selectbox("Select a document:", options["label"].tolist(), index=0)
    selected_id = selected_label.split(" — ")[0].strip()

    doc = df_view[df_view["document_id"].astype(str) == str(selected_id)].iloc[0]

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Prune Score", f"{float(doc['prune_score']):.2f}")
    with colB:
        st.metric("Shelf-Life", f"{float(doc['predicted_shelf_life_months']):.1f} mo")
    with colC:
        st.metric("Risk Level", str(doc["risk_level"]))
    with colD:
        st.metric("Last Access", f"{int(doc['last_access_days'])} days")

    st.markdown("---")
    explain_prune_score(doc, df_view)

    st.markdown("---")
    st.markdown("### 🎯 Demo Actions (mock)")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("✅ Keep", use_container_width=True):
            st.success("Marked as Keep (demo only).")
    with a2:
        if st.button("📦 Archive", use_container_width=True):
            st.info("Marked for Archive (demo only).")
    with a3:
        if st.button("🗑️ Review for Deletion", use_container_width=True):
            st.warning("Flagged for deletion review (demo only).")

# ----------------------------
# TABS
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "🌳 Knowledge Garden", "📋 Prune Candidates", "📊 Analytics", "🔍 Deep Dive"]
)

# TAB 1: OVERVIEW
with tab1:
    st.subheader("Document Landscape")

    fig = px.scatter(
        df_filtered,
        x="last_access_days",
        y="predicted_shelf_life_months",
        color="prune_score",
        size="citation_count",
        hover_data={
            "title": True,
            "publication_year": True,
            "access_count": True,
            "prune_score": ":.2f",
            "predicted_shelf_life_months": ":.1f",
        },
        color_continuous_scale="RdYlGn_r",
        labels={
            "last_access_days": "Days Since Last Access",
            "predicted_shelf_life_months": "Predicted Shelf-Life (months)",
            "prune_score": "Prune Score",
        },
        title="Document Relevance Landscape",
    )
    fig.update_layout(height=520)

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Interpretation**: Documents in the upper-left (recently accessed, long shelf-life) are core references. "
        "Bottom-right documents (old, short shelf-life) are prune candidates."
    )

# TAB 2: KNOWLEDGE GARDEN
with tab2:
    st.subheader("🌳 Your Knowledge Garden")
    st.caption("Visual metaphor: Tree trunk = frequently used, Leaves = prune candidates")

    df_viz = df_filtered.copy()
    df_viz["count"] = 1

    fig_sunburst = px.sunburst(
        df_viz,
        path=["risk_level", "field"],
        values="count",
        color="prune_score",
        color_continuous_scale="RdYlGn_r",
        title="Library Composition by Risk Level & Field",
    )
    fig_sunburst.update_layout(height=620)
    st.plotly_chart(fig_sunburst, use_container_width=True)

    st.markdown("**🍃 Leaf Colors:**")
    st.markdown("- 🟢 Green: Low prune score (keep)")
    st.markdown("- 🟡 Yellow: Medium prune score (review)")
    st.markdown("- 🔴 Red: High prune score (consider pruning)")

# TAB 3: PRUNE CANDIDATES
with tab3:
    st.subheader(f"Documents with Prune Score ≥ {prune_threshold:.2f}")

    if df_prune_candidates.empty:
        st.success("✅ No documents meet the pruning threshold. Your library is well-maintained!")
    else:
        st.warning(f"⚠️ {len(df_prune_candidates)} documents suggested for review")

        df_display = df_prune_candidates.sort_values("prune_score", ascending=False).copy()
        df_display["Prune Score"] = df_display["prune_score"].apply(lambda x: f"{x:.2f}")
        df_display["Shelf-Life"] = df_display["predicted_shelf_life_months"].apply(lambda x: f"{x:.1f} months")
        df_display["Last Access"] = df_display["last_access_days"].apply(
            lambda x: f"{int(x)} days ago" if pd.notna(x) else "n/a"
        )

        display_cols = [
            "document_id",
            "title",
            "publication_year",
            "field",
            "Last Access",
            "access_count",
            "Prune Score",
            "Shelf-Life",
            "risk_level",
        ]

        st.dataframe(df_display[display_cols], use_container_width=True, height=420)

        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Prune Candidates (CSV)",
            data=csv,
            file_name=f"prune_candidates_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# TAB 4: ANALYTICS
with tab4:
    st.subheader("📊 Library Analytics")

    colA, colB = st.columns(2)

    with colA:
        fig_hist = px.histogram(
            df_filtered,
            x="prune_score",
            nbins=30,
            title="Prune Score Distribution",
            labels={"prune_score": "Prune Score", "count": "Number of Documents"},
        )
        fig_hist.add_vline(
            x=prune_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({prune_threshold:.2f})",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with colB:
        fig_box = px.box(
            df_filtered,
            x="field",
            y="predicted_shelf_life_months",
            color="risk_level",
            title="Shelf-Life by Research Field",
            labels={"predicted_shelf_life_months": "Shelf-Life (months)"},
        )
        # Correct Plotly function is update_xaxes (plural)
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### 📅 Document Timeline")

    fig_timeline = px.scatter(
        df_filtered.sort_values("publication_year"),
        x="publication_year",
        y="last_access_days",
        color="prune_score",
        size="access_count",
        hover_data=["title", "predicted_shelf_life_months"],
        color_continuous_scale="RdYlGn_r",
        title="Publication Year vs. Access Recency",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# TAB 5: DEEP DIVE
with tab5:
    deep_dive_tab(df_filtered)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>
    Knowledge Garden v0.3 - Interim Prototype<br>
    Using mock predictions. Final version will integrate trained survival models (Cox PH, Random Survival Forest).<br>
    Target Metrics: C-index ≥ 0.70, IBS ≤ 0.18, SUS ≥ 75
    </small>
</div>
""",
    unsafe_allow_html=True,
)
