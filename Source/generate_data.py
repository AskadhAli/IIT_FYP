"""
generate_data.py
----------------
Generates a synthetic Knowledge Garden demo dataset for offline use / testing.

FIX (#17): N raised from 500 to 50,000 to match the survival model's design
scale. Generating only 500 documents produces statistically unreliable C-index
and IBS values.

FIX (#18): The `risk_level` column computed here is based on the heuristic
`prune_score`, not the ML model. app.py overwrites this column with
`ml_risk_level` after running the survival model, so the column generated here
is effectively unused in the main pipeline. A clear comment is added to
document this.

FIX (#5): Replaced global np.random.seed() with a local numpy Generator
(np.random.default_rng) to avoid polluting the global random state.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Use a local RNG — do NOT set np.random.seed() globally
rng = np.random.default_rng(42)

# FIX (#17): Raised from 500 to 50,000 to match model design scale
N = 50_000

topics = ["Machine Learning", "Human-Computer Interaction", "Data Mining",
          "Computer Vision", "Natural Language Processing", "Software Engineering",
          "Databases", "Networking", "Security", "AI Ethics"]

methods = ["Survey", "Framework", "Algorithm", "Analysis", "Study", "Review",
           "Approach", "System", "Model", "Tool"]

contexts = ["for Prediction", "in Education", "with Deep Learning",
            "for Healthcare", "in Social Media", "for Sustainability",
            "with Uncertainty", "at Scale", "in Practice", "and Beyond"]

def generate_title():
    topic   = rng.choice(topics)
    method  = rng.choice(methods)
    context = rng.choice(contexts)
    return f"{topic} {method} {context}"

data = {
    'document_id':       [f'DOC_{i:04d}' for i in range(1, N+1)],
    'title':             [generate_title() for _ in range(N)],
    'authors':           [f'Author{i} et al.' for i in range(N)],
    'publication_year':  rng.integers(2015, 2025, N),
    'field':             rng.choice(topics, N),
}

df = pd.DataFrame(data)

current_date = datetime(2026, 1, 14)

# Days since added (0-1095 days, ~3 years)
df['days_since_added'] = rng.exponential(scale=400, size=N).astype(int)
df['days_since_added'] = np.clip(df['days_since_added'], 0, 1095)

# Last access
recency_bias = 1 - (df['days_since_added'] / df['days_since_added'].max())
noise = rng.exponential(scale=60, size=N)
df['last_access_days'] = (df['days_since_added'] * 0.3 + noise * (1 - recency_bias)).astype(int)
df['last_access_days'] = np.clip(df['last_access_days'], 0, df['days_since_added'])

# Access count
df['access_count'] = rng.negative_binomial(n=2, p=0.4, size=N) + 1
df['access_count'] = np.clip(df['access_count'], 1, 50)

# Annotations
annotation_prob = np.clip(df['access_count'] / 20, 0.1, 0.9)
df['has_annotations']  = rng.binomial(1, annotation_prob)
df['annotation_count'] = np.where(
    df['has_annotations'] == 1,
    rng.poisson(lam=5, size=N),
    0
)

# Citations
age_years = (current_date.year - df['publication_year'])
df['citation_count'] = rng.poisson(lam=age_years * 3, size=N)
df['citation_count'] = np.clip(df['citation_count'], 0, 200)

# Derived features
df['recency_score']    = np.exp(-df['last_access_days'] / 90)
months_since_added     = df['days_since_added'] / 30
df['access_frequency'] = df['access_count'] / np.maximum(months_since_added, 1)

df['health_score'] = (
    0.4 * df['recency_score'] +
    0.3 * np.clip(df['access_frequency'], 0, 1) +
    0.2 * np.clip(df['has_annotations'], 0, 1) +
    0.1 * np.clip(df['citation_count'] / 50, 0, 1)
)

# Heuristic prune score (pre-ML)
df['prune_score'] = 1 - df['health_score'] + rng.normal(0, 0.1, N)
df['prune_score'] = np.clip(df['prune_score'], 0, 1)

# Shelf-life estimate
df['predicted_shelf_life_months'] = 24 * (1 - df['prune_score']) + rng.normal(0, 3, N)
df['predicted_shelf_life_months'] = np.clip(df['predicted_shelf_life_months'], 0, 36)

# Confidence intervals
df['shelf_life_ci_lower'] = np.maximum(0, df['predicted_shelf_life_months'] - rng.uniform(2, 6, N))
df['shelf_life_ci_upper'] = df['predicted_shelf_life_months'] + rng.uniform(2, 6, N)

df['tags'] = df['field'].apply(lambda x: f"{x};Research;Academic")

# FIX (#18): risk_level here is derived from the HEURISTIC prune_score, not the
# ML survival model. In app.py this column is overwritten by ml_risk_level after
# predict_on_dataframe() runs. It is kept here for reference / offline use only.
df['risk_level'] = pd.cut(df['prune_score'],
                          bins=[0, 0.3, 0.6, 1.0],
                          labels=['Low', 'Medium', 'High'])

output_df = df[[
    'document_id', 'title', 'authors', 'publication_year', 'field', 'tags',
    'days_since_added', 'last_access_days', 'access_count',
    'has_annotations', 'annotation_count', 'citation_count',
    'recency_score', 'access_frequency', 'health_score',
    'prune_score', 'predicted_shelf_life_months',
    'shelf_life_ci_lower', 'shelf_life_ci_upper', 'risk_level'
]]

numeric_cols = output_df.select_dtypes(include=[np.number]).columns
output_df[numeric_cols] = output_df[numeric_cols].round(2)

output_df.to_excel('Dataset/knowledge_garden_demo_v2.xlsx', index=False, engine='openpyxl')
print(f"Generated {N} realistic documents")
print(f"Prune score distribution:")
print(output_df['prune_score'].describe())
print(f"\nRisk level distribution (heuristic — overwritten by ML model in app):")
print(output_df['risk_level'].value_counts())
print(f"\nSaved to: Dataset/knowledge_garden_demo_v2.xlsx")