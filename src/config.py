"""
Central configuration for paths and constants across the project.

This module provides consistent absolute paths regardless of where
scripts are executed from. All modules should import paths from here
instead of calculating them individually.
"""

from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_DATABASES = DATA_DIR / "databases"
DATA_PREDICTIONS = DATA_DIR / "predictions"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Images directory
IMAGES_DIR = PROJECT_ROOT / "images"

# Database file
DATABASE_PATH = DATA_DATABASES / "fraud_detection_local.db"

# Data files
RAW_DATA_FILE = DATA_RAW / "data.json"
TRAIN_DATA_FILE = DATA_PROCESSED / "train_data.json"
TEST_DATA_FILE = DATA_PROCESSED / "test_data.json"
SIMULATE_DATA_FILE = DATA_PROCESSED / "simulate_data.json"

# Model files
MODEL_FOREST = MODELS_DIR / "forest_clf.pkl"
MODEL_NLP_NAME = MODELS_DIR / "nlp_name_text_clf_pipeline.pkl"
MODEL_NLP_DESC = MODELS_DIR / "nlp_description_text_clf_pipeline.pkl"
MODEL_NLP_ORG_NAME = MODELS_DIR / "nlp_org_name_text_clf_pipeline.pkl"
MODEL_NLP_ORG_DESC = MODELS_DIR / "nlp_org_desc_text_clf_pipeline.pkl"

# Flask app configuration
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Model constants
FRAUD_THRESHOLD_HIGH = 0.10
FRAUD_THRESHOLD_MEDIUM = 0.03
