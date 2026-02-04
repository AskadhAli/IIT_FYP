"""Test script to verify ArXiv data transformation"""
from load_arxiv_data import load_arxiv_dataset, transform_arxiv_to_knowledge_garden

# Test with a small sample
print("Testing ArXiv data transformation with 100 papers...")
df_arxiv = load_arxiv_dataset(sample_size=100)
print(f"\nLoaded {len(df_arxiv)} papers from ArXiv")

print("\nArXiv columns:", list(df_arxiv.columns))

# Transform
df_kg = transform_arxiv_to_knowledge_garden(df_arxiv)

print(f"\nTransformed to {len(df_kg)} documents")
print("\nKnowledge Garden columns:", list(df_kg.columns))

# Check required columns
required_cols = {
    "document_id", "title", "publication_year", "field",
    "last_access_days", "access_count", "citation_count",
    "prune_score", "predicted_shelf_life_months", "risk_level"
}

missing = required_cols - set(df_kg.columns)
if missing:
    print(f"\nERROR: Missing columns: {missing}")
else:
    print("\nSUCCESS: All required columns present!")

# Show sample
print("\nSample data:")
print(df_kg[['document_id', 'title', 'publication_year', 'field', 'prune_score', 'risk_level']].head())

print("\nField distribution:")
print(df_kg['field'].value_counts())

print("\nRisk level distribution:")
print(df_kg['risk_level'].value_counts())

print("\nTest complete!")
