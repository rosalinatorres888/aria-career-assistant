"""
ARIA Job Search Configuration
Target roles, keywords, and scoring criteria for Rosalina Torres
"""

TARGET_TITLES = [
    "data analyst", "data analytics engineer", "data engineer",
    "data scientist", "machine learning engineer", "ml engineer",
    "ai engineer", "ai/ml engineer", "analytics engineer",
    "business intelligence", "data analytics intern",
    "data science intern", "machine learning intern", "ml intern",
    "ai intern", "data engineering intern", "data analyst intern",
    "data science co-op", "data engineering co-op", "ml co-op",
]

POSITIVE_KEYWORDS = [
    "python", "sql", "machine learning", "data analytics",
    "pandas", "scikit-learn", "tensorflow", "pytorch",
    "nlp", "natural language processing", "deep learning",
    "statistics", "statistical learning", "data mining",
    "etl", "data pipeline", "airflow", "spark",
    "tableau", "power bi", "visualization",
    "a/b testing", "experimentation",
    "entry level", "junior", "associate", "new grad",
    "internship", "co-op", "summer 2026",
    "masters", "graduate", "ms student",
    "bilingual", "spanish",
]

NEGATIVE_KEYWORDS = [
    "senior", "staff", "principal", "director",
    "10+ years", "8+ years", "7+ years",
    "phd required", "doctorate required", "security clearance",
]

TARGET_LOCATIONS = [
    "boston", "massachusetts", "cambridge", "somerville",
    "remote", "hybrid", "flexible", "new york", "nyc",
]

EXCLUDED_COMPANIES = ["google", "alphabet", "amazon", "aws"]

SCORING = {
    "title_match_weight": 0.40,
    "keyword_match_weight": 0.25,
    "location_match_weight": 0.20,
    "recency_weight": 0.15,
    "negative_keyword_penalty": 0.30,
    "excluded_company_penalty": 1.00,
}
