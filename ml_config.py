"""
ml_config.py
------------
The ONE place that describes our dataset.

train.py and predict.py both read their settings from here.
Dataset: global_placement.csv (10,000 students, 13 columns).
"""

from pathlib import Path

# Folder that contains this file (salary-ai/)
BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# 1. DATASET
# ==================================================
DATASET_PATH = BASE_DIR / "data" / "global_placement.csv"


# ==================================================
# 2. INPUT FEATURES (the X columns)
# ==================================================
# Text columns. Converted to numbers with pd.get_dummies (one-hot),
# exactly as in the Colab analysis (untitled2.py).
CATEGORICAL_FEATURES = [
    "college_tier",
    "country",
    "university_ranking_band",
    "specialization",
    "industry",
]

# Number columns. Used as they are (Random Forest does not need scaling).
NUMERICAL_FEATURES = [
    "cgpa",
    "backlogs",
    "internship_count",
    "aptitude_score",
    "communication_score",
    "internship_quality_score",
]

# Number columns that must be whole numbers in the form
INTEGER_FEATURES = ["backlogs", "internship_count"]

# Friendly names shown on the website
FEATURE_LABELS = {
    "cgpa": "CGPA",
    "backlogs": "Backlogs",
    "college_tier": "College Tier",
    "country": "Country",
    "university_ranking_band": "University Ranking Band",
    "internship_count": "Internship Count",
    "aptitude_score": "Aptitude Score",
    "communication_score": "Communication Score",
    "specialization": "Specialization",
    "industry": "Industry",
    "internship_quality_score": "Internship Quality Score",
}

# Order of the fields in the web form
FORM_FIELDS = [
    "cgpa",
    "backlogs",
    "college_tier",
    "country",
    "university_ranking_band",
    "internship_count",
    "aptitude_score",
    "communication_score",
    "specialization",
    "industry",
    "internship_quality_score",
]


# ==================================================
# 3. TARGETS (the y columns)
# ==================================================
# Employment -> CLASSIFICATION (Placed / Not Placed)
EMPLOYMENT_TARGET = "placement_status"
POSITIVE_CLASS = "Placed"

# Salary -> REGRESSION (trained only on students who were placed)
SALARY_TARGET = "salary"


# ==================================================
# 4. TRAINING SETTINGS (same values as the Colab analysis)
# ==================================================
TEST_SIZE = 0.20      # 20% of rows are kept aside for testing
RANDOM_STATE = 42     # makes results repeatable
N_ESTIMATORS = 200    # number of trees in each Random Forest


# ==================================================
# 5. WHERE THE TRAINED MODELS ARE SAVED
# ==================================================
# Same file names as the Colab analysis, so its .pkl files can be
# copied straight into models/ as well.
MODELS_DIR = BASE_DIR / "models"
PLACEMENT_MODEL_PATH = MODELS_DIR / "placement_model.pkl"
SALARY_MODEL_PATH = MODELS_DIR / "salary_model.pkl"
PLACEMENT_COLUMNS_PATH = MODELS_DIR / "placement_columns.pkl"
SALARY_COLUMNS_PATH = MODELS_DIR / "salary_columns.pkl"
CATEGORICAL_COLS_PATH = MODELS_DIR / "categorical_cols.pkl"

# Written by train.py: category values, number ranges, metrics,
# feature importances. Used to build the form and the AI insights.
METADATA_PATH = MODELS_DIR / "model_metadata.json"


# ==================================================
# 6. DISPLAY
# ==================================================
# Salary currency symbol shown on the website.
# NOTE: check the dataset's actual currency before presenting.
CURRENCY_SYMBOL = "₹"
