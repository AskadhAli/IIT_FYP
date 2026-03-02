"""
Data transformation module to convert ArXiv dataset to Knowledge Garden format.

FIX (#19): simulate_access_metrics and simulate_citation_count previously used
the global numpy random state without a seed, making every run produce different
synthetic usage data and breaking reproducibility. All simulation functions now
accept and use a numpy Generator (np.random.default_rng) threaded through from
a single seeded instance at the top of transform_arxiv_to_knowledge_garden().

FIX (#20): extract_publication_year previously returned None silently when year
extraction failed, causing downstream NaN pub years to all be imputed to 2020.
It now returns a sentinel value (0) to flag these records so callers can decide
how to handle them, and the transform function logs the fraction affected.

FIX (#21): The unused KnowledgeGardenAccessor pandas accessor has been removed.
It registered a .kg namespace as a side effect on every import but was never
called anywhere in the codebase.
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

    FIX (#20): Previously returned None on failure. Now returns 0 as a
    sentinel so callers can distinguish "no year available" from a valid year,
    rather than silently treating all failures as 2020.

    Args:
        versions: List of version dicts with 'created' dates
        update_date: String date like "2008-11-26"

    Returns:
        Year as integer, or 0 if extraction failed.
    """
    # Try to get year from first version
    if versions and isinstance(versions, list) and len(versions) > 0:
        first_version = versions[0]
        if isinstance(first_version, dict) and 'created' in first_version:
            try:
                date_str = first_version['created']
                try:
                    from dateutil import parser
                    date = parser.parse(date_str)
                    return date.year
                except ImportError:
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                    if year_match:
                        return int(year_match.group())
            except Exception:
                pass

    # Fallback to update_date
    if update_date and pd.notna(update_date):
        try:
            date = pd.to_datetime(update_date)
            return date.year
        except Exception:
            pass

    # FIX (#20): Return 0 as explicit sentinel instead of None
    return 0


def map_category_to_field(categories):
    """
    Map ArXiv categories to research fields.
    """
    if pd.isna(categories):
        return "Other"

    categories_str = str(categories).lower()

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

    for prefix, field in category_map.items():
        if prefix in categories_str:
            return field

    first_cat = categories_str.split()[0] if ' ' in categories_str else categories_str
    if '.' in first_cat:
        main_cat = first_cat.split('.')[0]
        return main_cat.capitalize()

    return "Other"


def simulate_access_metrics(publication_year, current_year, rng):
    """
    Simulate access_count and last_access_days based on publication year.

    FIX (#19): Accepts a numpy Generator instance instead of using the global
    random state. This ensures reproducibility across runs.

    Args:
        publication_year: Year paper was published (0 = unknown)
        current_year: Current year
        rng: numpy.random.Generator instance

    Returns:
        Tuple of (access_count, last_access_days)
    """
    # FIX (#20): Handle sentinel value 0 (unknown year) gracefully
    if publication_year == 0 or pd.isna(publication_year):
        publication_year = 2015  # conservative default for unknown year

    age_years = current_year - publication_year

    base_access  = max(1, 50 - age_years * 2)
    access_count = int(rng.exponential(base_access))
    access_count = max(1, min(access_count, 200))

    if age_years < 1:
        last_access_days = rng.exponential(30)
    elif age_years < 3:
        last_access_days = rng.exponential(90)
    elif age_years < 5:
        last_access_days = rng.exponential(180)
    else:
        last_access_days = rng.exponential(365 * min(age_years / 2, 5))

    last_access_days = max(1, int(last_access_days))
    return access_count, last_access_days


def simulate_citation_count(publication_year, categories, current_year, rng):
    """
    Simulate citation count based on publication year and field.

    FIX (#19): Accepts a numpy Generator instance for reproducibility.

    Args:
        publication_year: Year paper was published (0 = unknown)
        categories: ArXiv categories
        current_year: Current year
        rng: numpy.random.Generator instance

    Returns:
        Citation count
    """
    # FIX (#20): Handle sentinel value 0 (unknown year)
    if publication_year == 0 or pd.isna(publication_year):
        publication_year = 2015

    age_years = current_year - publication_year

    if age_years <= 0:
        base_citations = 5
    elif age_years <= 2:
        base_citations = 10 + age_years * 5
    elif age_years <= 5:
        base_citations = 20 + (age_years - 2) * 3
    else:
        base_citations = 29 + (age_years - 5) * 1

    categories_str = str(categories).lower() if pd.notna(categories) else ""
    if 'cs.' in categories_str or 'stat.ml' in categories_str:
        multiplier = 1.5
    elif 'math.' in categories_str:
        multiplier = 0.8
    else:
        multiplier = 1.0

    citations = int(base_citations * multiplier * rng.lognormal(0, 0.5))
    return max(0, citations)


def calculate_prune_score(last_access_days, access_count, citation_count,
                          has_annotations, publication_year, current_year=2024):
    """
    Calculate prune score based on multiple factors.
    Higher score = more likely to prune.
    """
    recency_score = np.exp(-last_access_days / 90)
    access_score  = min(1.0, access_count / 10.0)
    citation_score = min(1.0, citation_count / 50.0)
    annotation_bonus = 0.2 if has_annotations > 0 else 0.0

    # FIX (#20): Handle sentinel value 0 for unknown year
    if publication_year == 0 or pd.isna(publication_year):
        publication_year = 2015
    age_years   = current_year - publication_year
    age_penalty = max(0, (age_years - 10) * 0.05) if age_years > 10 else 0

    health_score = (
        0.4 * recency_score +
        0.3 * access_score +
        0.2 * citation_score +
        0.1 * (1 - min(age_penalty, 0.5)) +
        annotation_bonus * 0.1
    )

    prune_score = np.clip(1 - health_score, 0.0, 1.0)
    return prune_score


def transform_arxiv_to_knowledge_garden(df_arxiv, current_year=2024, seed=42):
    """
    Transform ArXiv dataset to Knowledge Garden format.

    FIX (#19): A single seeded Generator is created here and threaded through
    all simulation functions, ensuring the transformation is fully reproducible.

    FIX (#20): Papers with extraction-failed years (sentinel value 0) are
    logged so the analyst can see the data quality issue.

    Args:
        df_arxiv: DataFrame with ArXiv data
        current_year: Current year for calculations
        seed: Random seed for reproducibility

    Returns:
        DataFrame in Knowledge Garden format
    """
    print("Transforming ArXiv data to Knowledge Garden format...")

    # FIX (#19): Single seeded Generator for all simulation calls
    rng = np.random.default_rng(seed)

    df = pd.DataFrame()

    df['document_id'] = df_arxiv['id'].astype(str)
    df['title']       = df_arxiv['title'].astype(str)

    print("Extracting publication years...")
    df['publication_year'] = df_arxiv.apply(
        lambda row: extract_publication_year(row.get('versions'), row.get('update_date')),
        axis=1
    )

    # FIX (#20): Log the fraction of records with unknown publication year
    unknown_year_count = (df['publication_year'] == 0).sum()
    if unknown_year_count > 0:
        pct = unknown_year_count / len(df) * 100
        print(f"  WARNING: {unknown_year_count} records ({pct:.1f}%) have unknown publication "
              f"year (extraction failed). These will use a conservative default of 2015 "
              f"for age-dependent features.")

    print("Mapping categories to research fields...")
    df['field'] = df_arxiv['categories'].apply(map_category_to_field)

    print("Simulating access metrics...")
    # FIX (#19): pass rng to each simulation call
    df_reset       = df.reset_index(drop=True)
    df_arxiv_reset = df_arxiv.reset_index(drop=True)

    access_metrics = [
        simulate_access_metrics(df_reset.loc[i, 'publication_year'], current_year, rng)
        for i in range(len(df_reset))
    ]
    df['access_count']     = [x[0] for x in access_metrics]
    df['last_access_days'] = [x[1] for x in access_metrics]

    print("Simulating citation counts...")
    df['citation_count'] = [
        simulate_citation_count(
            df_reset.loc[i, 'publication_year'],
            df_arxiv_reset.loc[i, 'categories'] if i < len(df_arxiv_reset) else None,
            current_year,
            rng,
        )
        for i in range(len(df_reset))
    ]

    # Annotations correlated with access_count
    df['has_annotations'] = (
        (df['access_count'] > 5).astype(int) *
        rng.binomial(1, 0.3, len(df))
    )
    df['annotation_count'] = df['has_annotations'] * rng.poisson(3, len(df))

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

    # Shelf-life estimate
    base_shelf_life = 24
    noise = rng.normal(0, 3, len(df))
    df['predicted_shelf_life_months'] = np.clip(
        base_shelf_life * (1 - df['prune_score']) + noise,
        0, 36
    )

    # Risk level (heuristic — overwritten by ML model in app.py)
    df['risk_level'] = pd.cut(
        df['prune_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )

    # Ensure numeric types
    for col in ['prune_score', 'predicted_shelf_life_months', 'last_access_days',
                'access_count', 'citation_count', 'has_annotations', 'annotation_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # FIX (#20): Convert sentinel 0 back to NaN for publication_year so
    # the column has correct nullable integer semantics (0 is not a valid year)
    df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
    df['publication_year'] = df['publication_year'].replace(0, pd.NA)
    df['publication_year'] = df['publication_year'].astype('Int64')

    print(f"Transformation complete! {len(df)} documents ready.")
    return df