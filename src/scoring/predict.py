"""
Event record processing for fraud detection.

This module contains the EventRecord class for processing individual fraud detection
records through the complete feature engineering and prediction pipeline.
"""

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src.data.preprocessing import convert_datetimes


class EventRecord:
    """
    Processes a single event record through the fraud detection pipeline.

    This class handles:
    - Loading raw event data
    - Data cleaning and validation
    - Feature engineering (aggregates, categorical encoding)
    - HTML to text conversion
    - NLP feature probability generation
    - Final fraud probability prediction

    Used by the Flask app's /score endpoint for real-time predictions.
    """

    def __init__(self, record_json=None):
        """
        Initialize EventRecord.

        Args:
            record_json (dict, optional): Raw event data from API or file.
                If provided, automatically loads the record.
        """
        self.record_id = None
        self.original_record = None
        self.processed_record = None
        self.nlp_predictions = {}
        self.final_prediction = None

        # Auto-load if record provided
        if record_json is not None:
            self.load_record_from_json(record_json)

    def load_record_from_json(self, record_json):
        """
        Load event record from JSON data.

        Args:
            record_json (dict): Raw event data from API or file
        """
        self.record_id = record_json["object_id"]
        self.original_record = record_json

    def clean_record(self):
        """
        Clean and engineer features using the validated training pipeline.

        This ensures production predictions match training/testing by using
        the same preprocessing logic from run_data_prep_pipeline().
        """
        from src.data.preprocessing import run_data_prep_pipeline

        # Convert single record to DataFrame for pipeline processing
        record_df = pd.DataFrame([self.original_record])

        # Apply the same preprocessing pipeline used in training
        processed_df = run_data_prep_pipeline(record_df)

        # Convert back to Series for compatibility with existing code
        self.processed_record = processed_df.iloc[0]

    def convert_html_to_text(self):
        """
        Convert HTML content in description fields to plain text.

        Note: This is now handled by run_data_prep_pipeline() in clean_record(),
        but keeping this method for backwards compatibility with existing code.
        """
        # HTML conversion is already done in the pipeline
        pass

    def create_nlp_predictions(self, trained_nlp_models):
        """
        Generate fraud probability predictions from text features.

        Args:
            trained_nlp_models (dict): Mapping of feature names to trained NLP pipelines
                                      (e.g., {'name': model, 'description': model, ...})

        Adds probability features to processed_record and stores in nlp_predictions.
        """
        self.nlp_predictions = {}
        for feature, model in trained_nlp_models.items():
            fraud_index = np.argwhere(model.classes_ == "Fraud")[0, 0]
            prediction = model.predict_proba([self.processed_record[feature]])[
                :, fraud_index
            ]
            prediction = round(float(prediction), 5)
            self.processed_record[f"{feature}_proba"] = prediction
            self.nlp_predictions[feature] = prediction

    def predict_record(self, trained_clf_model):
        """
        Generate final fraud probability prediction using the trained classifier.

        Args:
            trained_clf_model: Trained Random Forest classifier

        Sets final_prediction attribute with fraud probability (0-1).
        """
        features = [
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

        record_features = self.processed_record[features].to_frame().T
        fraud_index = 0  # Random Forest classes are ["Fraud", "Not Fraud"]
        prediction = trained_clf_model.predict_proba(record_features)[:, fraud_index]
        prediction = round(float(prediction), 5)

        self.final_prediction = prediction
        self.processed_record["fraud_proba"] = prediction

        # Add risk classification with threshold optimized for maximum fraud detection
        if (
            prediction >= 0.10
        ):  # Optimized threshold for 90% recall (prioritizing fraud detection)
            self.processed_record["risk_classification"] = "High"
        elif prediction >= 0.03:
            self.processed_record["risk_classification"] = "Medium"
        else:
            self.processed_record["risk_classification"] = "Low"
