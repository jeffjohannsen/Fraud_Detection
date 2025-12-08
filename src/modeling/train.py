#!/usr/bin/env python3
"""
Retrain all fraud detection models with current scikit-learn version.
This script retrains:
1. Four NLP text classification pipelines (name, description, org_name, org_desc)
2. Final Random Forest classifier

Run from project root: python src/modeling/train.py
"""

import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import MODELS_DIR
from src.data.preprocessing import load_fraud_data, run_data_prep_pipeline


def train_nlp_pipeline(X_train, y_train, feature_name):
    """
    Train an NLP pipeline for a text feature.

    Args:
        X_train: Training text data
        y_train: Training labels
        feature_name: Name of the feature being trained

    Returns:
        Trained pipeline
    """
    print(f"\n  Training NLP pipeline for '{feature_name}'...")

    text_clf_pipeline = Pipeline(
        [
            ("vect", CountVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                SGDClassifier(loss="modified_huber", max_iter=10000, random_state=42),
            ),
        ]
    )

    text_clf_pipeline.fit(X_train, y_train)

    # Calculate accuracy
    train_accuracy = text_clf_pipeline.score(X_train, y_train)
    print(f"    Training accuracy: {train_accuracy:.4f}")

    return text_clf_pipeline


def add_nlp_features(df, nlp_models):
    """
    Add NLP probability features to dataframe.

    Args:
        df: Dataframe with text features
        nlp_models: Dict of trained NLP models

    Returns:
        Dataframe with added probability features
    """
    for feature_name, model in nlp_models.items():
        fraud_index = np.argwhere(model.classes_ == "Fraud")[0, 0]
        predictions = model.predict_proba(df[feature_name])[:, fraud_index]
        df[f"{feature_name}_proba"] = predictions

    return df


def train_final_classifier(X_train, y_train, X_test, y_test):
    """
    Train the final Random Forest classifier.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Trained Random Forest model
    """
    print("\n  Training final Random Forest classifier...")

    # Use the same hyperparameters that worked well in original training
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    rf_clf.fit(X_train, y_train)

    train_accuracy = rf_clf.score(X_train, y_train)
    test_accuracy = rf_clf.score(X_test, y_test)

    print(f"    Training accuracy: {train_accuracy:.4f}")
    print(f"    Test accuracy: {test_accuracy:.4f}")

    return rf_clf


def save_model(model, filename):
    """Save model to models directory."""
    filepath = MODELS_DIR / f"{filename}.pkl"
    with open(filepath, "wb") as f:
        pkl.dump(model, f)
    print(f"    ✓ Saved to {filepath}")


def main():
    print("=" * 70)
    print("FRAUD DETECTION MODEL RETRAINING")
    print("=" * 70)

    # 1. Load and prepare data
    print("\n[1/3] Loading and preparing data...")
    print("  Loading train_data.json...")
    original_data = load_fraud_data("train_data")

    print("  Running data preparation pipeline...")
    df = run_data_prep_pipeline(original_data)

    print(f"  ✓ Loaded {len(df):,} records")
    fraud_count = (df["is_fraud"] == "Fraud").sum()
    print(
        f"  ✓ Fraud rate: {fraud_count/len(df)*100:.1f}% ({fraud_count:,} fraud records)"
    )

    # 2. Train NLP pipelines
    print("\n[2/3] Training NLP text classification pipelines...")

    text_features = ["name", "description", "org_name", "org_desc"]
    nlp_models = {}

    for feature in text_features:
        # Split data for this feature
        X_train, X_test, y_train, y_test = train_test_split(
            df[feature],
            df["is_fraud"],
            test_size=0.25,
            random_state=42,
            stratify=df["is_fraud"],
        )

        # Train model
        model = train_nlp_pipeline(X_train, y_train, feature)
        nlp_models[feature] = model

        # Save model
        save_model(model, f"nlp_{feature}_text_clf_pipeline")

    # 3. Add NLP features and train final classifier
    print("\n[3/3] Training final classifier...")

    print("  Adding NLP probability features...")
    df = add_nlp_features(df, nlp_models)

    # Select features for final model
    final_features = [
        "body_length",
        "channels",
        "delivery_method",
        "fb_published",
        "gts",
        "has_analytics",
        "has_header",
        "has_logo",
        "listed",
        "name_length",
        "num_order",
        "num_payouts",
        "org_facebook",
        "org_twitter",
        "user_type",
        "sale_duration",
        "sale_duration2",
        "show_map",
        "user_age",
        "venue_latitude",
        "venue_longitude",
        "num_previous_payouts",
        "previous_payouts_total",
        "num_ticket_types",
        "num_tickets_available",
        "total_ticket_value",
        "avg_ticket_cost",
        "known_payee_name",
        "known_venue_name",
        "known_payout_type",
        "total_empty_values",
        "name_proba",
        "description_proba",
        "org_name_proba",
        "org_desc_proba",
    ]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df[final_features],
        df["is_fraud"],
        test_size=0.25,
        random_state=42,
        stratify=df["is_fraud"],
    )

    # Train Random Forest
    rf_model = train_final_classifier(X_train, y_train, X_test, y_test)
    save_model(rf_model, "forest_clf")

    print("\n" + "=" * 70)
    print("✅ MODEL RETRAINING COMPLETE!")
    print("=" * 70)
    print("\nAll models have been retrained with current scikit-learn version.")
    print("You can now run the Flask app: python run_app.py")
    print()


if __name__ == "__main__":
    main()
