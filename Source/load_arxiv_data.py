"""
Data transformation module to convert ArXiv dataset to Knowledge Garden format
"""
import kagglehub
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


def load_arxiv_dataset(sample_size=None, use_cache=True):
    """
    Load ArXiv dataset from Kaggle.
    
    Args:
        sample_size: If provided, only load this many rows (for testing)
        use_cache: Use cached dataset if available
    
    Returns:
        DataFrame with arxiv data
    """
    print("Loading ArXiv dataset...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    
    json_file = os.path.join(path, "arxiv-metadata-oai-snapshot.json")
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"ArXiv data file not found at {json_file}")
    
    # Load JSON file (it's JSONL format - one JSON object per line)
    print(f"Reading data from {json_file}...")
    
    data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} papers from ArXiv dataset")
    return df


def extract_publication_year(versions, update_date):
    """
    Extract publication year from versions or update_date.
    
    Args:
        versions: List of version dicts with 'created' dates
        update_date: String date like "2008-11-26"
    
    Returns:
        Year as integer or None
    """
    # Try to get year from first version
    if versions and isinstance(versions, list) and len(versions) > 0:
        first_version = versions[0]
        if isinstance(first_version, dict) and 'created' in first_version:
            try:
                date_str = first_version['created']
                # Parse date like "Mon, 2 Apr 2007 19:18:42 GMT"
                try:
                    from dateutil import parser
                    date = parser.parse(date_str)
                    return date.year
                except ImportError:
                    # Fallback: try to extract year from string
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                    if year_match:
                        return int(year_match.group())
                    return None
            except:
                pass
    
    # Fallback to update_date
    if update_date and pd.notna(update_date):
        try:
            # Parse date like "2008-11-26"
            date = pd.to_datetime(update_date)
            return date.year
        except:
            pass
    
    return None


def map_category_to_field(categories):
    """
    Map ArXiv categories to research fields.
    
    Args:
        categories: String like "hep-ph" or "math.CO cs.CG"
    
    Returns:
        Mapped field name
    """
    if pd.isna(categories):
        return "Other"
    
    categories_str = str(categories).lower()
    
    # Category mapping
    category_map = {
        'cs.': 'Computer Science',
        'math.': 'Mathematics',
        'physics.': 'Physics',
        'astro-ph': 'Astrophysics',
        'cond-mat': 'Condensed Matter',
        'hep-': 'High Energy Physics',
        'quant-ph': 'Quantum Physics',
        'q-bio': 'Quantitative Biology',
        'q-fin': 'Quantitative Finance',
        'stat.': 'Statistics',
        'econ.': 'Economics',
        'eess': 'Electrical Engineering',
        'gr-qc': 'General Relativity',
        'math-ph': 'Mathematical Physics',
        'nlin': 'Nonlinear Sciences',
    }
    
    # Check for matches
    for prefix, field in category_map.items():
        if prefix in categories_str:
            return field
    
    # Default mapping based on first category
    first_cat = categories_str.split()[0] if ' ' in categories_str else categories_str
    if '.' in first_cat:
        main_cat = first_cat.split('.')[0]
        return main_cat.capitalize()
    
    return "Other"


def simulate_access_metrics(publication_year, current_year=2024):
    """
    Simulate access_count and last_access_days based on publication year.
    Older papers are accessed less frequently.
    
    Args:
        publication_year: Year paper was published
        current_year: Current year
    
    Returns:
        Tuple of (access_count, last_access_days)
    """
    if pd.isna(publication_year):
        publication_year = 2020  # Default
    
    age_years = current_year - publication_year
    
    # Older papers: fewer accesses, longer since last access
    # Newer papers: more accesses, recent access
    
    # Access count: decreases with age, with some randomness
    base_access = max(1, 50 - age_years * 2)
    access_count = int(np.random.exponential(base_access))
    access_count = max(1, min(access_count, 200))  # Cap between 1 and 200
    
    # Last access days: increases with age
    if age_years < 1:
        # Very recent: accessed recently
        last_access_days = np.random.exponential(30)
    elif age_years < 3:
        # Recent: accessed within last year
        last_access_days = np.random.exponential(90)
    elif age_years < 5:
        # Medium age: accessed within last 2 years
        last_access_days = np.random.exponential(180)
    else:
        # Old: accessed long ago
        last_access_days = np.random.exponential(365 * min(age_years / 2, 5))
    
    last_access_days = max(1, int(last_access_days))
    
    return access_count, last_access_days


def simulate_citation_count(publication_year, categories, current_year=2024):
    """
    Simulate citation count based on publication year and field.
    Some fields (like ML/AI) have higher citation rates.
    
    Args:
        publication_year: Year paper was published (can be None)
        categories: ArXiv categories
        current_year: Current year
    
    Returns:
        Citation count
    """
    if pd.isna(publication_year) or publication_year is None:
        publication_year = 2020
    
    age_years = current_year - publication_year
    
    # Base citations increase with time (up to a point)
    if age_years <= 0:
        base_citations = 5
    elif age_years <= 2:
        base_citations = 10 + age_years * 5
    elif age_years <= 5:
        base_citations = 20 + (age_years - 2) * 3
    else:
        base_citations = 29 + (age_years - 5) * 1
    
    # Field multiplier (CS/AI papers get more citations)
    categories_str = str(categories).lower() if pd.notna(categories) else ""
    if 'cs.' in categories_str or 'stat.ml' in categories_str:
        multiplier = 1.5
    elif 'math.' in categories_str:
        multiplier = 0.8
    else:
        multiplier = 1.0
    
    citations = int(base_citations * multiplier * np.random.lognormal(0, 0.5))
    return max(0, citations)


def calculate_prune_score(last_access_days, access_count, citation_count, 
                          has_annotations, publication_year, current_year=2024):
    """
    Calculate prune score based on multiple factors.
    Higher score = more likely to prune.
    
    Args:
        last_access_days: Days since last access
        access_count: Number of times accessed
        citation_count: Number of citations
        has_annotations: Whether document has annotations
        publication_year: Year published
        current_year: Current year
    
    Returns:
        Prune score between 0 and 1
    """
    # Recency score (exponential decay)
    recency_score = np.exp(-last_access_days / 90)
    
    # Access frequency score
    access_score = min(1.0, access_count / 10.0)
    
    # Citation score
    citation_score = min(1.0, citation_count / 50.0)
    
    # Annotation bonus
    annotation_bonus = 0.2 if has_annotations > 0 else 0.0
    
    # Age penalty (very old papers)
    age_years = current_year - (publication_year if pd.notna(publication_year) else 2020)
    age_penalty = max(0, (age_years - 10) * 0.05) if age_years > 10 else 0
    
    # Health score (higher = better, lower = prune)
    health_score = (
        0.4 * recency_score +
        0.3 * access_score +
        0.2 * citation_score +
        0.1 * (1 - min(age_penalty, 0.5)) +
        annotation_bonus * 0.1
    )
    
    # Prune score is inverse of health
    prune_score = 1 - health_score
    prune_score = np.clip(prune_score, 0.0, 1.0)
    
    return prune_score


def transform_arxiv_to_knowledge_garden(df_arxiv, current_year=2024):
    """
    Transform ArXiv dataset to Knowledge Garden format.
    
    Args:
        df_arxiv: DataFrame with ArXiv data
        current_year: Current year for calculations
    
    Returns:
        DataFrame in Knowledge Garden format
    """
    print("Transforming ArXiv data to Knowledge Garden format...")
    
    df = pd.DataFrame()
    
    # Map basic fields
    df['document_id'] = df_arxiv['id'].astype(str)
    df['title'] = df_arxiv['title'].astype(str)
    
    # Extract publication year
    print("Extracting publication years...")
    df['publication_year'] = df_arxiv.apply(
        lambda row: extract_publication_year(row.get('versions'), row.get('update_date')),
        axis=1
    )
    
    # Map categories to fields
    print("Mapping categories to research fields...")
    df['field'] = df_arxiv['categories'].apply(map_category_to_field)
    
    # Simulate access metrics
    print("Simulating access metrics...")
    access_metrics = df.apply(
        lambda row: simulate_access_metrics(row['publication_year'], current_year),
        axis=1
    )
    df['access_count'] = [x[0] for x in access_metrics]
    df['last_access_days'] = [x[1] for x in access_metrics]
    
    # Simulate citation counts
    print("Simulating citation counts...")
    # Reset index to align with df_arxiv
    df_reset = df.reset_index(drop=True)
    df_arxiv_reset = df_arxiv.reset_index(drop=True)
    df['citation_count'] = [
        simulate_citation_count(
            pub_year,
            df_arxiv_reset.loc[i, 'categories'] if i < len(df_arxiv_reset) else None,
            current_year
        )
        for i, pub_year in enumerate(df_reset['publication_year'])
    ]
    
    # Simulate annotations (random, but correlated with access_count)
    df['has_annotations'] = (df['access_count'] > 5).astype(int) * np.random.binomial(1, 0.3, len(df))
    df['annotation_count'] = df['has_annotations'] * np.random.poisson(3, len(df))
    
    # Calculate prune score
    print("Calculating prune scores...")
    df['prune_score'] = df.apply(
        lambda row: calculate_prune_score(
            row['last_access_days'],
            row['access_count'],
            row['citation_count'],
            row['has_annotations'],
            row['publication_year'],
            current_year
        ),
        axis=1
    )
    
    # Calculate predicted shelf-life (inverse of prune score)
    base_shelf_life = 24  # months
    df['predicted_shelf_life_months'] = base_shelf_life * (1 - df['prune_score']) + np.random.normal(0, 3, len(df))
    df['predicted_shelf_life_months'] = np.clip(df['predicted_shelf_life_months'], 0, 36)
    
    # Risk level based on prune score
    df['risk_level'] = pd.cut(
        df['prune_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # Ensure numeric types
    for col in ['prune_score', 'predicted_shelf_life_months', 'last_access_days', 
                'access_count', 'citation_count', 'has_annotations', 'annotation_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce').astype('Int64')
    
    print(f"Transformation complete! {len(df)} documents ready.")
    return df


@pd.api.extensions.register_dataframe_accessor("kg")
class KnowledgeGardenAccessor:
    """Custom accessor for Knowledge Garden data operations"""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def load_arxiv(self, sample_size=None):
        """Load and transform ArXiv data"""
        df_arxiv = load_arxiv_dataset(sample_size=sample_size)
        return transform_arxiv_to_knowledge_garden(df_arxiv)
