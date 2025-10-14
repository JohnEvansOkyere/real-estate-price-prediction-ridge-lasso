"""
Configuration file for house price prediction project.
All constants, paths, and parameters in one place.
"""

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "datasets"          # Changed from "data" to "datasets"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Output directories
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Create output directories
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
TRAIN_CLEANED_PATH = PROCESSED_DATA_DIR / "train_cleaned.csv"
MODEL_PATH = MODEL_DIR / "ridge_model.pkl"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = REPORTS_DIR / "metrics.csv"
REPORT_PATH = REPORTS_DIR / "model_report.txt"

# ============================================================
# DATA CLEANING
# ============================================================
# Outlier bounds for numerical features
SQUARE_FT_MIN = 100
SQUARE_FT_MAX = 10000
BHK_MIN = 1
BHK_MAX = 10

# Price outlier removal (IQR multiplier)
PRICE_IQR_MULTIPLIER = 3

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Categorical feature mappings
POSTED_BY_MAPPING = {'Builder': 'Dealer'}  # Merge Builder into Dealer

# Features to drop
DROP_FEATURES = ['BHK_OR_RK', 'ADDRESS']

# Geographic clustering
N_GEO_CLUSTERS = 20

# ============================================================
# MODEL TRAINING
# ============================================================
# Random state for reproducibility
RANDOM_STATE = 42

# Train/validation split
TEST_SIZE = 0.2

# Feature groups
NUMERIC_FEATURES = ['SQUARE_FT', 'BHK_NO.', 'sqft_per_bhk', 'log_sqft', 
                   'LATITUDE', 'LONGITUDE']
BINARY_FEATURES = ['UNDER_CONSTRUCTION', 'RERA', 'READY_TO_MOVE', 'RESALE']
CATEGORICAL_FEATURES = ['POSTED_BY', 'city', 'geo_cluster']

# Hyperparameter search grids
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
LASSO_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1, 10]

# Cross-validation folds
CV_FOLDS = 5

# ============================================================
# TARGET VARIABLE
# ============================================================
TARGET_COL = 'TARGET(PRICE_IN_LACS)'

# Use log transformation
USE_LOG_TRANSFORM = True