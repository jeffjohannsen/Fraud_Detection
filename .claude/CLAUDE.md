# Claude Instructions for Fraud Detection Project

## Project Overview

This is a **production-ready fraud detection system** built for event transaction monitoring. It uses machine learning with NLP feature engineering to classify events as fraudulent or legitimate. The system includes a Flask web application for real-time predictions and historical analysis.

**Key characteristics:**
- **Binary classification**: "Fraud" vs "Not Fraud" (~9% fraud rate - imbalanced dataset)
- **Dual-threshold approach**: 92.9% recall, prioritizing fraud detection over false positives
- **Local deployment**: SQLite database, Flask web app (development/demo system)
- **Full ML pipeline**: Data preprocessing → NLP feature engineering → Random Forest classifier

This is a **portfolio project** - focus on code quality, reproducibility, and production-ready patterns.

## Environment Setup

**CRITICAL:** Always use the virtual environment. Never use system Python.

```bash
# Use venv Python directly (recommended for terminal commands)
./venv/bin/python script.py

# Or activate venv first (for interactive sessions)
source venv/bin/activate  # bash/zsh
```

All dependencies are installed in `./venv/`.

## Architecture & Data Flow

```
Raw Data (JSON) → Preprocessing → NLP Feature Engineering → ML Models
                                                                ↓
                     Simulation → EventRecord → Predictions → Database
                                                                ↓
                                             Flask Web App (User Interface)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Data preprocessing | `src/data/preprocessing.py` | Cleaning, feature engineering, type conversion |
| NLP features | `src/data/nlp_features.py` | Text→fraud probability pipelines |
| Model training | `src/modeling/train.py` | Train/retrain all models |
| Prediction engine | `src/scoring/predict.py` | `EventRecord` class for scoring |
| Flask webapp | `src/webapp/app.py` | Real-time predictions, dashboard |
| Database adapter | `src/database/db.py` | SQLite interface via `LocalDatabase` class |

## Data Conventions

### Data Directory Structure

```
data/
├── raw/               # Original unprocessed datasets
│   └── data.json     # Full dataset: 14,337 records (gitignored - in GitHub Release)
├── processed/         # Chronologically split datasets
│   ├── train_data.json      # 8,602 records (60%, 2007-mid-2013, 9.6% fraud)
│   ├── test_data.json       # 2,867 records (20%, mid-2013, 4.4% fraud)
│   └── simulate_data.json   # 2,868 records (20%, late-2013, 12.0% fraud) - included in repo
├── databases/         # SQLite database files (gitignored except demo db)
└── predictions/       # Model outputs (gitignored)
```

**Chronological split rationale:** Train/test/simulate are split by time to mimic real-world deployment where models predict future fraud.

### Data Loading Pattern

```python
from src.data.preprocessing import load_fraud_data

# Load processed datasets
df_train = load_fraud_data("train_data")      # Training set
df_test = load_fraud_data("test_data")        # Test/validation set
df_simulate = load_fraud_data("simulate_data") # Webapp demo set
```

### Target Variable

- **Column name**: `is_fraud`
- **Values**: `"Fraud"` or `"Not Fraud"` (strings, NOT binary 0/1)
- **Class distribution**: ~9% fraud overall (varies by split)

### Missing Value Handling

**Never drop rows with missing values** - missingness is predictive of fraud!

- **Empty strings** → `"Unknown"` (for text/categorical features)
- **Null numerics** → `-1` (sentinel value)
- **Engineered feature**: `total_empty_values` counts unknowns per record

### Complex Features

**Nested structures** require aggregation:
- `previous_payouts` (list of dicts) → `num_previous_payouts`, `total_payout_amount`, etc.
- `ticket_types` (list of dicts) → `num_ticket_types`, `avg_ticket_cost`, `total_ticket_value`, etc.

**HTML features** require text extraction:
- `description` (HTML) → plain text via BeautifulSoup
- `org_desc` (HTML) → plain text via BeautifulSoup

```python
from bs4 import BeautifulSoup
clean_text = BeautifulSoup(html, "html.parser").get_text("|", strip=True)
```

## Model Conventions

### Model Files

Stored as pickle files in `models/` directory:

```python
# NLP text classification pipelines (4 models)
models/nlp_name_text_clf_pipeline.pkl
models/nlp_description_text_clf_pipeline.pkl
models/nlp_org_name_text_clf_pipeline.pkl
models/nlp_org_desc_text_clf_pipeline.pkl

# Final classifier
models/forest_clf.pkl  # Random Forest with 36 features
```

### NLP Feature Engineering

Four text features are transformed into fraud probabilities using sklearn pipelines:

**Architecture**: `CountVectorizer → TfidfTransformer → SGDClassifier`

| Original Feature | NLP Pipeline Output | Notes |
|-----------------|---------------------|-------|
| `name` | `name_proba` | Event name text |
| `description` | `description_proba` | Event description (HTML→text) |
| `org_name` | `org_name_proba` | Organization name |
| `org_desc` | `org_desc_proba` | Organization description (HTML→text) |

### Final Classifier Features (36 total)

**Categories:**
1. **Numeric features**: `body_length`, `channels`, `gts`, `user_age`, `sale_duration`, `fb_published`, `has_analytics`, `has_logo`, etc.
2. **Engineered features**: `num_ticket_types`, `avg_ticket_cost`, `known_payee_name`, `total_empty_values`, `country` (categorical encoding)
3. **NLP probabilities**: `name_proba`, `description_proba`, `org_name_proba`, `org_desc_proba`

### Dual-Threshold Classification System

**CRITICAL - This is how the production system works:**

The Random Forest outputs fraud probabilities that are classified into three risk levels, but only two matter for prediction:

| Risk Level | Probability Range | Production Behavior |
|-----------|-------------------|---------------------|
| **High** | ≥ 0.10 | **Flagged as fraud** |
| **Medium** | ≥ 0.03 and < 0.10 | **Flagged as fraud** |
| **Low** | < 0.03 | Not flagged |

**Effective threshold**: 0.03 (both High and Medium are treated as fraud predictions)

**Performance metrics** (on test set):
- **Recall**: 92.9% (catches 118 out of 127 fraudulent events)
- **Precision**: 25.7% (1 in 4 flagged events is actually fraud)
- **False Positive Rate**: 12.4% (1 in 8 legitimate events flagged)

**Rationale**: Prioritize catching nearly all fraud over minimizing false alarms. Flagged events undergo manual review before action is taken.

**Technical note**: Random Forest `classes_` array is `['Fraud', 'Not Fraud']`, so fraud probability is at **index 0** (not 1).

### Metrics & Evaluation

**Primary metric**: **Recall** (maximize fraud detection)
**Secondary metrics**: Accuracy, Precision, F1-Score, False Positive Rate

**Baseline performance** (predict all "Not Fraud"):
- Accuracy: ~90% (due to class imbalance)
- Recall: 0% (catches no fraud)

**Model performance**:
- **Logistic Regression**: 95% accuracy, 52% recall
- **Random Forest (dual-threshold)**: 95.9% accuracy, 92.9% recall ✅

## Database

### SQLite Setup

- **Database file**: `data/databases/fraud_detection_local.db`
- **Connection string**: `sqlite:///data/databases/fraud_detection_local.db`
- **Table**: `fraud_records`
- **Demo data**: Included in repository with 1,800 historical predictions

### Database Schema

The `fraud_records` table stores predictions with:
- Event details (name, organization, country, etc.)
- Fraud probability (0.0 to 1.0)
- Risk classification (High/Medium/Low)
- Timestamp

### Database Management

```python
from src.database.db import LocalDatabase

# Initialize database connection
db = LocalDatabase("data/databases/fraud_detection_local.db")

# Verify database status
# Run: python src/database/init.py
```

## Running the System

### Flask Web Application

```bash
# From project root
./venv/bin/python run_app.py

# Access at: http://127.0.0.1:5000
```

**Available endpoints:**
- `/` - Home page with recent predictions dashboard
- `/simulate` - Simulate fraud detection on random events from holdout data
- `/score` - API endpoint for scoring events (POST JSON)

### Database Initialization

```bash
# Verify/initialize database
./venv/bin/python src/database/init.py
```

### Model Retraining

```bash
# Retrain all models (4 NLP pipelines + Random Forest)
./venv/bin/python src/modeling/train.py
```

This retrains models using current scikit-learn version and saves to `models/` directory.

## Project Structure

```
Fraud_Detection/
├── data/
│   ├── raw/                        # Original datasets (gitignored)
│   ├── processed/                  # Train/test/simulate splits
│   ├── databases/                  # SQLite database
│   └── predictions/                # Model outputs (gitignored)
├── models/                         # Trained models (.pkl files)
│   ├── nlp_*_text_clf_pipeline.pkl # 4 NLP pipelines
│   └── forest_clf.pkl              # Random Forest classifier
├── notebooks/
│   ├── exploratory_data_analysis.ipynb  # EDA with visualizations
│   ├── model_development.ipynb          # Model training & comparison
│   └── wordcloud_creation.ipynb         # Text feature visualization
├── src/
│   ├── data/
│   │   ├── preprocessing.py        # Data cleaning & feature engineering
│   │   ├── nlp_features.py         # NLP text processing pipelines
│   │   └── holdout.py              # Train/test/simulate split utilities
│   ├── modeling/
│   │   └── train.py                # Model retraining script
│   ├── database/
│   │   ├── db.py                   # SQLite database adapter
│   │   └── init.py                 # Database initialization script
│   ├── scoring/
│   │   └── predict.py              # EventRecord class for predictions
│   └── webapp/
│       ├── app.py                  # Flask application
│       ├── static/                 # CSS, JS, images
│       └── templates/              # HTML templates (Jinja2)
├── scripts/                        # Utility scripts
├── images/                         # Visualizations for README
├── run_app.py                      # Flask app entry point
├── requirements.txt                # Python dependencies
└── README.md
```

## Code Style & Patterns

### Import Pattern in Webapp

Flask app modifies `sys.path` to import from parent directory:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now can import from src/
from src.scoring.predict import EventRecord
```

### Docstrings

Use **NumPy style** for functions in `src/` modules.

### Type Hints

Use when it improves clarity, especially for function signatures.

### Testing Pattern

Use `simulate_data.json` for webapp demonstrations (included in repo, not gitignored).

## Important Notes & Caveats

### Data Availability

**Full datasets NOT in git** (too large):
- Download from [GitHub Release](https://github.com/jeffjohannsen/Fraud_Detection/releases)
- `data.json` (239 MB) → `data/raw/`
- `train_data.json` (164 MB) → `data/processed/`
- `test_data.json` (44 MB) → `data/processed/`
- `simulate_data.json` IS included in repo for webapp demo

### Model Assumptions & Limitations

1. **Class imbalance**: ~9% fraud rate requires careful threshold selection
2. **IID violations**: Multiple events from same user/organization violate independence assumptions
3. **Temporal bias**: Model trained on 2007-2013 data may not generalize to future fraud patterns
4. **Geographic bias**: Training data heavily weighted toward US/UK events

### Known Technical Quirks

1. **String target variable**: `is_fraud` is `"Fraud"/"Not Fraud"` strings (not 0/1)
2. **Random Forest class order**: `classes_[0] = 'Fraud'`, so fraud probability is `predict_proba[:, 0]`
3. **Missing value encoding**: `-1` and `"Unknown"` are intentional, don't "fix" them
4. **HTML in data**: `description` and `org_desc` contain HTML that must be parsed with BeautifulSoup

## Common Tasks

### Adding a New Feature

1. Add feature engineering logic to `src/data/preprocessing.py` → `run_data_prep_pipeline()`
2. Update feature list in `src/modeling/train.py`
3. Retrain models: `./venv/bin/python src/modeling/train.py`
4. Test with webapp simulation

### Debugging Predictions

1. Check feature values in `EventRecord` object
2. Verify NLP probabilities are in [0, 1] range
3. Check Random Forest feature importance
4. Review threshold logic in `predict.py`

### Modifying Thresholds

Edit `src/scoring/predict.py` → `EventRecord.classify_risk()` method:

```python
if fraud_prob >= 0.10:
    return "High"
elif fraud_prob >= 0.03:  # Adjust this threshold
    return "Medium"
else:
    return "Low"
```

Retest recall/precision tradeoff after changes.
