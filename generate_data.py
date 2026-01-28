import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of documents to generate
N = 500

# Generate realistic academic paper titles
topics = ["Machine Learning", "Human-Computer Interaction", "Data Mining", 
          "Computer Vision", "Natural Language Processing", "Software Engineering",
          "Databases", "Networking", "Security", "AI Ethics"]

methods = ["Survey", "Framework", "Algorithm", "Analysis", "Study", "Review",
           "Approach", "System", "Model", "Tool"]

contexts = ["for Prediction", "in Education", "with Deep Learning", 
            "for Healthcare", "in Social Media", "for Sustainability",
            "with Uncertainty", "at Scale", "in Practice", "and Beyond"]

def generate_title():
    topic = np.random.choice(topics)
    method = np.random.choice(methods)
    context = np.random.choice(contexts)
    return f"{topic} {method} {context}"

# Generate base data
data = {
    'document_id': [f'DOC_{i:04d}' for i in range(1, N+1)],
    'title': [generate_title() for _ in range(N)],
    'authors': [f'Author{i} et al.' for i in range(N)],
    'publication_year': np.random.randint(2015, 2025, N),
    'field': np.random.choice(topics, N),
}

df = pd.DataFrame(data)

# Calculate derived features with realistic distributions
current_date = datetime(2026, 1, 14)  # Your project date

# Days since added (0-1095 days, ~3 years)
df['days_since_added'] = np.random.exponential(scale=400, size=N).astype(int)
df['days_since_added'] = np.clip(df['days_since_added'], 0, 1095)

# Last access follows different patterns based on document age
# Newer docs accessed more recently
recency_bias = 1 - (df['days_since_added'] / df['days_since_added'].max())
noise = np.random.exponential(scale=60, size=N)
df['last_access_days'] = (df['days_since_added'] * 0.3 + noise * (1 - recency_bias)).astype(int)
df['last_access_days'] = np.clip(df['last_access_days'], 0, df['days_since_added'])

# Access count (realistic: most papers accessed 1-5 times, few accessed heavily)
df['access_count'] = np.random.negative_binomial(n=2, p=0.4, size=N) + 1
df['access_count'] = np.clip(df['access_count'], 1, 50)

# Has annotations (more likely if accessed more)
annotation_prob = np.clip(df['access_count'] / 20, 0.1, 0.9)
df['has_annotations'] = np.random.binomial(1, annotation_prob)

# Annotation count (if has_annotations)
df['annotation_count'] = np.where(
    df['has_annotations'] == 1,
    np.random.poisson(lam=5, size=N),
    0
)

# Citation count (older papers tend to have more, but noisy)
age_years = (current_date.year - df['publication_year'])
df['citation_count'] = np.random.poisson(lam=age_years * 3, size=N)
df['citation_count'] = np.clip(df['citation_count'], 0, 200)

# Calculate features that would predict relevance decay
# Recency score (exponential decay)
df['recency_score'] = np.exp(-df['last_access_days'] / 90)

# Access frequency (accesses per month since added)
months_since_added = df['days_since_added'] / 30
df['access_frequency'] = df['access_count'] / np.maximum(months_since_added, 1)

# Document "health" score (composite indicator)
df['health_score'] = (
    0.4 * df['recency_score'] + 
    0.3 * np.clip(df['access_frequency'], 0, 1) +
    0.2 * np.clip(df['has_annotations'], 0, 1) +
    0.1 * np.clip(df['citation_count'] / 50, 0, 1)
)

# GENERATE REALISTIC PRUNE SCORES
# High prune score = likely to be obsolete
# Based on: old last access, low access frequency, no annotations
df['prune_score'] = 1 - df['health_score'] + np.random.normal(0, 0.1, N)
df['prune_score'] = np.clip(df['prune_score'], 0, 1)

# PREDICTED SHELF-LIFE (months remaining of usefulness)
# Inverse relationship with prune score
# Documents with low prune scores have longer shelf-life
base_shelf_life = 24  # months
df['predicted_shelf_life_months'] = base_shelf_life * (1 - df['prune_score']) + np.random.normal(0, 3, N)
df['predicted_shelf_life_months'] = np.clip(df['predicted_shelf_life_months'], 0, 36)

# Add confidence intervals (simulate model uncertainty)
df['shelf_life_ci_lower'] = np.maximum(0, df['predicted_shelf_life_months'] - np.random.uniform(2, 6, N))
df['shelf_life_ci_upper'] = df['predicted_shelf_life_months'] + np.random.uniform(2, 6, N)

# Add tags/categories
df['tags'] = df['field'].apply(lambda x: f"{x};Research;Academic")

# Risk level based on prune score
df['risk_level'] = pd.cut(df['prune_score'], 
                          bins=[0, 0.3, 0.6, 1.0], 
                          labels=['Low', 'Medium', 'High'])

# Select final columns in logical order
output_df = df[[
    'document_id', 'title', 'authors', 'publication_year', 'field', 'tags',
    'days_since_added', 'last_access_days', 'access_count', 
    'has_annotations', 'annotation_count', 'citation_count',
    'recency_score', 'access_frequency', 'health_score',
    'prune_score', 'predicted_shelf_life_months', 
    'shelf_life_ci_lower', 'shelf_life_ci_upper', 'risk_level'
]]

# Round numeric columns
numeric_cols = output_df.select_dtypes(include=[np.number]).columns
output_df[numeric_cols] = output_df[numeric_cols].round(2)

# Save to Excel
output_df.to_excel('knowledge_garden_demo_v2.xlsx', index=False, engine='openpyxl')
print(f"✅ Generated {N} realistic documents")
print(f"📊 Prune score distribution:")
print(output_df['prune_score'].describe())
print(f"\n📈 Risk level distribution:")
print(output_df['risk_level'].value_counts())
print(f"\n💾 Saved to: knowledge_garden_demo_v2.xlsx")