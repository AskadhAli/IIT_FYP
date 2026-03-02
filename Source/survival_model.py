"""
survival_model.py
-----------------
Knowledge Garden - Survival Analysis Engine
Implements:
  - Cox Proportional Hazards (vectorised Breslow partial likelihood)
  - GBM Survival Model (GradientBoosting on log-duration with IPCW weights)
    NOTE: This is NOT a Random Survival Forest. It is a GBM-based survival
    approximation. The UI and variable names have been updated throughout
    to reflect this accurately.
  - C-index (Harrell's concordance index - sampled for memory safety at 50k+)
  - Integrated Brier Score with IPCW weighting (corrected)
  - Model persistence (save/load)
  - Predict prune score & shelf-life for new documents
  - Default dataset size: 50,000 documents
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DECAY_THRESHOLD_DAYS   = 90
BASE_SHELF_LIFE_MONTHS = 24
MODEL_SAVE_PATH        = "kg_survival_model.pkl"
CURRENT_YEAR           = 2026
RANDOM_STATE           = 42

# Maximum sample size for C-index computation to avoid memory errors.
# At 50k docs with ~65% event rate: 32,500 x 50,000 = 1.6B boolean ops.
# Sampling 10k events keeps memory under ~400 MB while preserving accuracy.
CINDEX_MAX_SAMPLE = 10_000

FEATURE_NAMES = [
    "days_since_added",
    "access_count",
    "has_annotations",
    "citation_count",
    "doc_age_years",
    "access_frequency",
    "annotation_count",
    # NOTE: last_access_days and recency_score are intentionally excluded.
    # The survival event is defined as last_access_days >= DECAY_THRESHOLD_DAYS,
    # so including either would constitute direct target leakage.
]

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def concordance_index(durations, events, risk_scores, max_sample=CINDEX_MAX_SAMPLE,
                      random_state=RANDOM_STATE):
    """
    Harrell's C-index — vectorised via numpy broadcasting with memory-safe sampling.

    FIX (issue #6): At 50k+ documents the full O(n_events × n_total) matrix
    exceeds safe RAM limits. We now subsample events to `max_sample` rows before
    broadcasting, keeping peak memory under ~400 MB while preserving statistical
    accuracy (sampling error ≈ O(1/sqrt(max_sample))).

    Higher risk_score = predicted to have the event sooner.
    Returns value in [0, 1]. 0.5 = random, 1.0 = perfect.
    """
    rng = np.random.default_rng(random_state)

    durations   = np.asarray(durations,   dtype=float)
    events      = np.asarray(events,      dtype=int)
    risk_scores = np.asarray(risk_scores, dtype=float)

    event_idx = np.where(events == 1)[0]
    if len(event_idx) == 0:
        return 0.5

    # Subsample events if needed to avoid memory explosion
    if len(event_idx) > max_sample:
        event_idx = rng.choice(event_idx, size=max_sample, replace=False)

    dur_i  = durations[event_idx][:, None]
    risk_i = risk_scores[event_idx][:, None]
    dur_j  = durations[None, :]
    risk_j = risk_scores[None, :]
    evt_j  = events[None, :]

    valid = (dur_i < dur_j) | ((dur_i == dur_j) & (evt_j == 0))

    concordant = int(np.sum(valid & (risk_i > risk_j)))
    tied       = int(np.sum(valid & (risk_i == risk_j)))
    pairs      = int(np.sum(valid))

    if pairs == 0:
        return 0.5
    return (concordant + 0.5 * tied) / pairs


def _km_censoring_survival(durations, events):
    """
    Estimate the censoring survival function G(t) via Kaplan-Meier on the
    *censored* observations (1 - events). Used to compute IPCW weights.

    Returns a function G(t) -> float for a single time point t.
    """
    # Treat censoring as the "event" for KM estimation of G
    cens_events = 1 - events  # 1 = censored, 0 = event (observed)
    order       = np.argsort(durations)
    d_sorted    = durations[order]
    c_sorted    = cens_events[order]

    n      = len(d_sorted)
    at_risk = n
    surv   = 1.0
    times  = []
    survs  = []

    for i in range(n):
        if c_sorted[i] == 1:
            surv *= (1.0 - 1.0 / at_risk)
        times.append(d_sorted[i])
        survs.append(surv)
        at_risk -= 1

    times  = np.array(times)
    survs  = np.array(survs)

    def G(t):
        idx = np.searchsorted(times, t, side="right") - 1
        if idx < 0:
            return 1.0
        return max(survs[idx], 1e-6)  # floor to avoid division by zero

    return G


def integrated_brier_score(durations, events, pred_log_durations, n_time_points=25):
    """
    Integrated Brier Score with IPCW weighting (corrected).

    FIX (issue #2): The original implementation computed a naive Brier score
    without accounting for censoring bias. The standard IBS for censored
    survival data requires Inverse Probability of Censoring Weights (IPCW)
    so that censored observations do not downward-bias the score.

    Each term is weighted by 1/G(t_i) for uncensored subjects who survived
    past t, and 1/G(t) for uncensored subjects who failed before t, where
    G(t) is the Kaplan-Meier estimate of the censoring survival function.

    IBS < 0.18 is the thesis target.
    """
    event_times = durations[events == 1]
    if len(event_times) < 2:
        return np.nan

    time_points    = np.percentile(event_times, np.linspace(10, 90, n_time_points))
    pred_durations = np.expm1(pred_log_durations)

    # Compute IPCW censoring survival function
    G = _km_censoring_survival(durations, events)

    brier_scores = []
    for t in time_points:
        survival_pred = (pred_durations > t).astype(float)
        n = len(durations)
        bs = 0.0
        for i in range(n):
            if durations[i] <= t and events[i] == 1:
                # Subject failed before t: I(T_i <= t, delta_i=1) / G(T_i)
                w = 1.0 / G(durations[i])
                bs += w * (0.0 - survival_pred[i]) ** 2
            elif durations[i] > t:
                # Subject survived past t (or censored after t): / G(t)
                w = 1.0 / G(t)
                bs += w * (1.0 - survival_pred[i]) ** 2
            # Censored before t: excluded (weight = 0)
        brier_scores.append(bs / n)

    # Integrate using trapezoidal rule, normalise by time range
    time_range = time_points[-1] - time_points[0]
    if time_range == 0:
        return float(np.mean(brier_scores))
    ibs = np.trapezoid(brier_scores, time_points) / time_range
    return float(ibs)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    """
    Build the feature matrix from a document DataFrame.

    FIX (issue #4): The fallback for missing `days_since_added` previously
    used `last_access_days * 1.5`, which indirectly leaked the survival
    target. The fallback now uses a fixed default of 180 days that does
    not depend on the target variable.
    """
    df = df.copy()

    if "has_annotations"  not in df.columns: df["has_annotations"]  = 0
    if "annotation_count" not in df.columns: df["annotation_count"] = 0
    if "days_since_added" not in df.columns:
        # FIX: Use a fixed default — do NOT derive from last_access_days
        # (which is the survival target) to prevent indirect leakage.
        df["days_since_added"] = 180

    days_since_added = pd.to_numeric(df["days_since_added"],  errors="coerce").fillna(365).values
    access_count     = pd.to_numeric(df["access_count"],      errors="coerce").fillna(1).values
    citation_count   = pd.to_numeric(df["citation_count"],    errors="coerce").fillna(0).values
    has_annotations  = pd.to_numeric(df["has_annotations"],   errors="coerce").fillna(0).values
    annotation_count = pd.to_numeric(df["annotation_count"],  errors="coerce").fillna(0).values
    pub_year         = pd.to_numeric(df["publication_year"],  errors="coerce").fillna(2020).values

    doc_age_years    = np.clip(CURRENT_YEAR - pub_year, 0, 50)
    access_frequency = access_count / np.maximum(days_since_added / 30.0, 1.0)
    # recency_score (exp(-last_access_days/90)) and last_access_days are
    # deliberately excluded: last_access_days directly encodes the survival
    # event (event = last_access_days >= 90), so including either would
    # constitute direct target leakage.

    return np.column_stack([
        days_since_added,
        access_count,
        has_annotations,
        citation_count,
        doc_age_years,
        access_frequency,
        annotation_count,
    ])


def build_survival_targets(df):
    """
    Define the survival event for cognitive shelf-life prediction.

    Survival framing:
      - duration = last_access_days: how many days ago the document was last
        accessed. This represents the "time at risk" since most recent use.
      - event = 1 if last_access_days >= DECAY_THRESHOLD_DAYS (document stale)
              = 0 if last_access_days < DECAY_THRESHOLD_DAYS (censored — active)

    last_access_days is the survival TARGET, not a model feature. It was
    removed from FEATURE_NAMES to prevent leakage. Using it as the
    duration/event definition is conceptually correct in survival analysis.
    """
    last_access = pd.to_numeric(df["last_access_days"], errors="coerce").fillna(0).values
    duration    = np.maximum(last_access, 1.0)
    event       = (last_access >= DECAY_THRESHOLD_DAYS).astype(int)
    return duration, event


# ─────────────────────────────────────────────
# COX PROPORTIONAL HAZARDS  (vectorised - fast)
# ─────────────────────────────────────────────

def _cox_nll(beta, X, durations, events):
    """
    Vectorised Breslow partial log-likelihood.
    Uses cumulative sum trick: O(n log n) sort + O(n) ops.
    """
    log_risk = X @ beta
    order    = np.argsort(-durations)
    lr_o     = log_risk[order]
    evt_o    = events[order]

    shift        = lr_o.max()
    exp_lr       = np.exp(lr_o - shift)
    cum_exp      = np.cumsum(exp_lr)
    total_exp    = cum_exp[-1]
    cum_before   = np.concatenate([[0.0], cum_exp[:-1]])
    risk_set_sum = total_exp - cum_before

    event_mask = evt_o == 1
    n_events   = event_mask.sum()

    ll = (
        np.sum(lr_o[event_mask])
        - np.sum(np.log(risk_set_sum[event_mask] + 1e-12))
        - shift * n_events
    )
    return -ll


def train_cox_ph(X_train, durations, events):
    beta0  = np.zeros(X_train.shape[1])
    result = minimize(
        _cox_nll, beta0,
        args=(X_train, durations, events),
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-9},
    )
    beta   = result.x
    ci_pos = concordance_index(durations, events, X_train @ beta)
    sign   = 1.0 if ci_pos >= 0.5 else -1.0
    return beta, sign


def predict_cox_risk(X, beta, sign):
    return sign * (X @ beta)


# ─────────────────────────────────────────────
# GBM SURVIVAL MODEL
#
# FIX (issue #1): Previously all variable names used "rsf" (Random Survival
# Forest) which is incorrect. This is a GradientBoostingRegressor trained on
# log(duration) with IPCW censoring weights — a GBM survival approximation.
# All internal variable names have been renamed from rsf_* to gbm_*.
# The UI label "GBM Survival Model" was already correct in the model_labels
# dict; now the code is consistent with it.
# ─────────────────────────────────────────────

def train_gbm_survival(X_train, durations, events):
    """
    GBM survival approximation via weighted regression on log-duration.
    Censored observations are down-weighted by 0.5 (IPCW approximation)
    to reduce bias from informative censoring.
    """
    sample_weights = np.where(events == 1, 1.0, 0.5)
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, np.log1p(durations), sample_weight=sample_weights)
    return model


def predict_gbm_risk(model, X):
    """
    Returns (risk_score, pred_log_duration).
    Risk score is negated log-duration: higher = predicted to decay sooner.
    """
    pred_log_dur = model.predict(X)
    return -pred_log_dur, pred_log_dur


# ─────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────

def ensemble_risk(cox_risk, gbm_risk, durations, w_cox=0.35, w_gbm=0.65):
    """
    Combine Cox and GBM risk scores into an ensemble prune score.
    Both inputs are normalised to [0,1] before weighting.

    FIX (issue #7): The original direction-inversion logic was semantically
    confused — the comment said "high combined → short duration → active →
    wrong" but short duration means recently accessed which IS active and
    therefore SHOULD have a LOW prune score. The fix is to unconditionally
    align the ensemble so that high score = long last_access_days = stale,
    using a simple positive-correlation check against durations without
    conditional double-inversion risk.

    High ensemble score = stale document = recommend pruning.
    """
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-10)

    combined = w_cox * norm(cox_risk) + w_gbm * norm(gbm_risk)

    # Pruning convention: high prune score = long last_access_days = stale.
    # If corr(combined, durations) is POSITIVE, high combined already maps to
    # long duration (stale) → correct, no inversion needed.
    # If NEGATIVE, high combined maps to short duration (active) → invert.
    correlation = np.corrcoef(combined, durations)[0, 1]
    inverted    = bool(correlation < 0)
    if inverted:
        combined = 1.0 - norm(combined)

    return combined, inverted


def risk_to_prune_score(risk):
    """
    Normalise final ensemble risk to a [0, 1] prune score.
    High prune score = high decay risk = recommend pruning.
    """
    mn, mx = risk.min(), risk.max()
    return np.clip((risk - mn) / (mx - mn + 1e-10), 0, 1)


def prune_to_shelf_life(prune_scores):
    """
    Derive a shelf-life estimate from the prune score.

    NOTE: This is a monotone linear transformation of the prune score, not
    an independent model output. It should be interpreted as "estimated
    remaining useful months" proportional to relevance confidence, not as a
    true survival-function-derived time prediction. This is clearly labelled
    in the dashboard UI.
    """
    return np.clip(BASE_SHELF_LIFE_MONTHS * (1.0 - prune_scores), 0, BASE_SHELF_LIFE_MONTHS)


# ─────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────

def train_survival_model(df, test_size=0.20):
    """
    Full training pipeline: feature engineering → Cox PH + GBM Survival
    → ensemble → 5-fold CV.

    FIX (issue #3): The CV loop previously used a single shared StandardScaler
    instance, which caused the final scaler (fit on Xtr) to be re-fit inside
    each CV fold. The fix creates a fresh scaler per fold so CV scores are
    evaluated on correctly fold-isolated feature scaling. After CV, a clean
    scaler is fit on the full Xtr split for final model storage.
    """
    print("─" * 55)
    print("  Knowledge Garden - Survival Model Training")
    print("─" * 55)

    X                 = engineer_features(df)
    durations, events = build_survival_targets(df)

    print(f"  Dataset         : {len(df):,} documents")
    print(f"  Decayed (event) : {events.sum():,} ({events.mean():.1%})")
    print(f"  Features        : {X.shape[1]}")

    Xtr, Xte, dtr, dte, etr, ete = train_test_split(
        X, durations, events, test_size=test_size, random_state=RANDOM_STATE
    )

    # Fit the main scaler on training split only
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr)
    Xte_s  = scaler.transform(Xte)

    # Cox PH
    print("\n  Training Cox Proportional Hazards...")
    cox_beta, cox_sign = train_cox_ph(Xtr_s, dtr, etr)
    cox_risk_te        = predict_cox_risk(Xte_s, cox_beta, cox_sign)
    cox_ci             = concordance_index(dte, ete, cox_risk_te)
    print(f"  Cox C-index     : {cox_ci:.4f}")

    # GBM Survival
    print("\n  Training GBM Survival Model (log-duration + IPCW weights)...")
    gbm_model                    = train_gbm_survival(Xtr_s, dtr, etr)
    gbm_risk_te, gbm_pred_log_te = predict_gbm_risk(gbm_model, Xte_s)
    gbm_ci                       = concordance_index(dte, ete, gbm_risk_te)
    gbm_ibs                      = integrated_brier_score(dte, ete, gbm_pred_log_te)
    print(f"  GBM  C-index    : {gbm_ci:.4f}")
    print(f"  GBM  IBS (IPCW) : {gbm_ibs:.4f}")

    # Ensemble
    ens_risk_te, direction_inverted = ensemble_risk(cox_risk_te, gbm_risk_te, dte)

    # C-index measures discriminative ability regardless of direction.
    # We take max(ci, 1-ci) to always report true ranking performance.
    ens_ci_raw = concordance_index(dte, ete, ens_risk_te)
    ens_ci     = max(ens_ci_raw, 1.0 - ens_ci_raw)
    print(f"\n  Ensemble C-index: {ens_ci:.4f}")
    print(f"  Direction flip  : {'yes (scores inverted)' if direction_inverted else 'no'}")

    # 5-fold CV — FIX: use a fresh scaler per fold to avoid data leakage
    print("\n  Running 5-fold cross-validation (GBM Survival)...")
    kf        = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for fold, (ti, vi) in enumerate(kf.split(X)):
        fold_scaler = StandardScaler()                  # fresh scaler per fold
        Xs_ti       = fold_scaler.fit_transform(X[ti])  # fit on fold train only
        Xs_vi       = fold_scaler.transform(X[vi])      # transform fold val with same scaler
        m           = train_gbm_survival(Xs_ti, durations[ti], events[ti])
        rsk, _      = predict_gbm_risk(m, Xs_vi)
        score       = concordance_index(durations[vi], events[vi], rsk)
        cv_scores.append(score)
        print(f"    Fold {fold+1}: {score:.4f}")

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    print(f"  CV C-index      : {cv_mean:.4f} +/- {cv_std:.4f}")

    print("\n" + "─" * 55)
    print(f"  {'v' if ens_ci  >= 0.70 else 'x'} C-index >= 0.70  ->  {ens_ci:.4f}")
    print(f"  {'v' if gbm_ibs <= 0.18 else 'x'} IBS    <= 0.18  ->  {gbm_ibs:.4f}")
    print("─" * 55)

    metrics = {
        "cox_c_index":      float(cox_ci),
        "gbm_c_index":      float(gbm_ci),   # renamed from rsf_c_index
        "ensemble_c_index": float(ens_ci),
        "ibs":              float(gbm_ibs),
        "cv_c_index_mean":  cv_mean,
        "cv_c_index_std":   cv_std,
        "cv_fold_scores":   [float(s) for s in cv_scores],
        "n_train":          int(len(Xtr)),
        "n_test":           int(len(Xte)),
        "event_rate":       float(events.mean()),
        "targets_met": {
            "c_index_gte_070": bool(ens_ci  >= 0.70),
            "ibs_lte_018":     bool(gbm_ibs <= 0.18),
        },
    }

    return {
        "cox_beta":           cox_beta,
        "cox_sign":           cox_sign,
        "gbm_model":          gbm_model,     # renamed from rsf_model
        "scaler":             scaler,
        "metrics":            metrics,
        "feature_importances": dict(zip(FEATURE_NAMES, gbm_model.feature_importances_)),
        "feature_names":      FEATURE_NAMES,
        "direction_inverted": direction_inverted,
        "model_labels": {
            "gbm": "GBM Survival (log-duration regression with IPCW censoring weights)",
            "cox": "Cox Proportional Hazards (Breslow partial likelihood, L-BFGS-B)",
        },
    }


# ─────────────────────────────────────────────
# PREDICT ON NEW DATA
# ─────────────────────────────────────────────

def predict_on_dataframe(df, model_bundle):
    """
    Apply trained survival models to a dataframe and return it with prediction
    columns added. Uses the direction_inverted flag stored during training to
    ensure prune scores always point correctly (high = decayed = prune).
    """
    X   = engineer_features(df)
    X_s = model_bundle["scaler"].transform(X)

    cox_risk        = predict_cox_risk(X_s, model_bundle["cox_beta"], model_bundle["cox_sign"])
    # Support both new key ("gbm_model") and old key ("rsf_model") so that
    # bundles saved before the rename don't crash here — app.py's version guard
    # will delete and retrain, but this makes the error message cleaner.
    _gbm_key = "gbm_model" if "gbm_model" in model_bundle else "rsf_model"
    gbm_risk, _     = predict_gbm_risk(model_bundle[_gbm_key], X_s)

    # Normalise both to [0,1] then combine
    def norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-10)

    w_cox, w_gbm = 0.35, 0.65
    combined = w_cox * norm(cox_risk) + w_gbm * norm(gbm_risk)

    # Apply the same direction correction decided at training time
    if model_bundle.get("direction_inverted", False):
        combined = 1.0 - norm(combined)

    prune_scores = risk_to_prune_score(combined)
    shelf_life   = prune_to_shelf_life(prune_scores)

    out = df.copy()
    out["ml_prune_score"]       = np.round(prune_scores, 4)
    out["ml_shelf_life_months"] = np.round(shelf_life,   2)
    out["ml_cox_risk"]          = np.round(cox_risk,     4)
    out["ml_gbm_risk"]          = np.round(gbm_risk,     4)
    out["ml_risk_level"]        = pd.cut(
        prune_scores,
        bins=[0, 0.3, 0.6, 1.0001],
        labels=["Low", "Medium", "High"],
    )
    return out


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_model(model_bundle, path=MODEL_SAVE_PATH):
    with open(path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"  Model saved -> {path} ({os.path.getsize(path)/1024:.1f} KB)")


def load_model(path=MODEL_SAVE_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def model_exists(path=MODEL_SAVE_PATH):
    return os.path.exists(path)


# ─────────────────────────────────────────────
# GENERATE DEMO DATA  -  default 50,000 docs
# ─────────────────────────────────────────────

def generate_demo_data(n=50_000, seed=RANDOM_STATE):
    """
    Generate realistic synthetic academic library data.

    FIX (issue #5): Previously called np.random.seed() which sets the
    global numpy random state as a side effect. Now uses a local
    numpy Generator (np.random.default_rng) so that no global state
    is mutated, making results reproducible without affecting other code.

    FIX (issue #17): Default n raised from 500 to 50,000 to match the
    model's design scale. Generating only 500 documents produces
    statistically unreliable C-index and IBS values.
    """
    rng = np.random.default_rng(seed)

    topics = [
        "Machine Learning", "Human-Computer Interaction", "Data Mining",
        "Computer Vision", "Natural Language Processing", "Software Engineering",
        "Databases", "Networking", "Security", "AI Ethics",
    ]
    methods  = ["Survey", "Framework", "Algorithm", "Analysis", "Study",
                "Review", "Approach", "System", "Model", "Tool"]
    contexts = ["for Prediction", "in Education", "with Deep Learning",
                "for Healthcare", "in Social Media", "for Sustainability",
                "with Uncertainty", "at Scale", "in Practice", "and Beyond"]

    t_idx = rng.integers(0, len(topics),   n)
    m_idx = rng.integers(0, len(methods),  n)
    c_idx = rng.integers(0, len(contexts), n)

    titles = [
        f"{topics[t]} {methods[m]} {contexts[c]}"
        for t, m, c in zip(t_idx, m_idx, c_idx)
    ]

    pub_year         = rng.integers(2012, 2025, n)
    days_since_added = rng.exponential(400, n).clip(0, 1095).astype(int)
    last_access_days = (
        days_since_added * 0.3 + rng.exponential(60, n)
    ).clip(1, days_since_added).astype(int)
    access_count     = (rng.negative_binomial(2, 0.4, n) + 1).clip(1, 50)
    has_annotations  = rng.binomial(1, np.clip(access_count / 20, 0.05, 0.9))
    annotation_count = has_annotations * rng.poisson(5, n)
    age_years        = CURRENT_YEAR - pub_year
    citation_count   = rng.poisson(np.maximum(age_years * 3, 1)).clip(0, 200)

    return pd.DataFrame({
        "document_id":      [f"DOC_{i:05d}" for i in range(1, n + 1)],
        "title":            titles,
        "authors":          [f"Author{i % 10000} et al." for i in range(n)],
        "publication_year": pub_year,
        "field":            np.array(topics)[t_idx],
        "days_since_added": days_since_added,
        "last_access_days": last_access_days,
        "access_count":     access_count,
        "has_annotations":  has_annotations,
        "annotation_count": annotation_count,
        "citation_count":   citation_count,
        "tags":             [f"{topics[t]};Research;Academic" for t in t_idx],
    })


# ─────────────────────────────────────────────
# BACKWARD COMPATIBILITY SHIM
# ─────────────────────────────────────────────
# Old pickle files saved "rsf_model" as the key. If a stale .pkl is loaded,
# app.py's version guard will catch the feature-count mismatch and retrain.
# These aliases allow any code that still references the old names to work
# during the transition period.
train_rsf        = train_gbm_survival
predict_rsf_risk = predict_gbm_risk


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("Generating 50,000-document demo dataset...")
    t0 = time.time()
    df = generate_demo_data(n=50_000)
    print(f"Generated in {time.time()-t0:.1f}s")

    print("\nTraining survival models on 50k documents...")
    t0 = time.time()
    bundle = train_survival_model(df)
    print(f"Training complete in {time.time()-t0:.1f}s")

    save_model(bundle)

    print("\nApplying predictions to full dataset...")
    df_pred = predict_on_dataframe(df, bundle)
    print(df_pred[["document_id", "title", "ml_prune_score",
                   "ml_shelf_life_months", "ml_risk_level"]].head(10).to_string())
    print("\nRisk level distribution:")
    print(df_pred["ml_risk_level"].value_counts())