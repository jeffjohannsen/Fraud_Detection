import json
import pickle as pkl
import random
from pathlib import Path

import flask

from src.config import DATA_PROCESSED, FLASK_DEBUG, FLASK_HOST, FLASK_PORT, MODELS_DIR
from src.database.db import LocalDatabase
from src.scoring.predict import EventRecord

# Initialize local database (uses DATABASE_PATH from config by default)
db = LocalDatabase()

# Country code to name mapping (ISO 3166-1 alpha-2)
COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "AU": "Australia",
    "NZ": "New Zealand",
    "DE": "Germany",
    "IE": "Ireland",
    "NL": "Netherlands",
    "VN": "Vietnam",
    "FR": "France",
    "PH": "Philippines",
    "BE": "Belgium",
    "HU": "Hungary",
    "PR": "Puerto Rico",
    "ES": "Spain",
    "RO": "Romania",
    "CH": "Switzerland",
    "SG": "Singapore",
    "MA": "Morocco",
    "IT": "Italy",
    "SE": "Sweden",
    "IN": "India",
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    "JP": "Japan",
    "CN": "China",
    "KR": "South Korea",
    "TH": "Thailand",
    "MY": "Malaysia",
    "ID": "Indonesia",
    "A1": "Anonymous Proxy",
    "": "Unknown",
}


def get_country_name(code):
    """Convert country code to full name."""
    return COUNTRY_NAMES.get(code, code)


# Load trained models
def load_models():
    """Load the trained NLP and classification models."""
    models = {}
    model_files = {
        "nlp_model_description": "nlp_description_text_clf_pipeline",
        "nlp_model_name": "nlp_name_text_clf_pipeline",
        "nlp_model_org_desc": "nlp_org_desc_text_clf_pipeline",
        "nlp_model_org_name": "nlp_org_name_text_clf_pipeline",
        "final_model": "forest_clf",
    }
    for key, filename in model_files.items():
        model_path = MODELS_DIR / f"{filename}.pkl"
        with open(model_path, "rb") as f:
            models[key] = pkl.load(f)
    return models


models = load_models()


def get_nlp_models():
    """Get NLP models dictionary for predictions."""
    return {
        "name": models["nlp_model_name"],
        "description": models["nlp_model_description"],
        "org_name": models["nlp_model_org_name"],
        "org_desc": models["nlp_model_org_desc"],
    }


def score_event_record(raw_record):
    """Score a single event record through the prediction pipeline.

    Args:
        raw_record (dict): Raw event data

    Returns:
        tuple: (record_id, fraud_probability, EventRecord object)
    """
    record = EventRecord(raw_record)
    record.clean_record()
    record.create_nlp_predictions(get_nlp_models())
    record.predict_record(models["final_model"])

    fraud_prob = round(record.final_prediction * 100, 1)
    return record.record_id, fraud_prob, record


def query_records_table():
    """Query database for all records and risk statistics."""
    risk_stats = db.get_risk_counts()
    records = db.get_all_records()
    return {**risk_stats, "records": records}


# Flask App Setup
webapp_dir = Path(__file__).parent
APP = flask.Flask(
    __name__,
    template_folder=str(webapp_dir / "templates"),
    static_folder=str(webapp_dir / "static"),
)


@APP.route("/")
def home():
    data = query_records_table()
    # Convert country codes to full names for display
    records_with_names = []
    for record in data["records"]:
        record_list = list(record)
        record_list[2] = get_country_name(record_list[2])  # Convert country code
        records_with_names.append(tuple(record_list))
    data["records"] = records_with_names
    return flask.render_template("home.html", **data)


@APP.route("/score", methods=["POST"])
def score():
    """Score a fraud record via POST request.

    Expects JSON payload with event data.
    Returns fraud probability and saves to database if new.
    """
    try:
        raw_record = flask.request.get_json()
        if not raw_record or "object_id" not in raw_record:
            return (
                flask.jsonify(
                    {
                        "status": "error",
                        "message": "Invalid or missing JSON data. 'object_id' field required.",
                    }
                ),
                400,
            )

        # Score the record
        r_id, r_pred, record = score_event_record(raw_record)

        # Check if already exists
        if db.record_exists(r_id):
            return flask.jsonify(
                {
                    "status": "exists",
                    "message": f"Record {r_id} already scored (fraud probability: {r_pred}%)",
                    "record_id": r_id,
                    "fraud_probability": r_pred,
                }
            )

        # Save new record to database
        record_dict = record.processed_record.to_dict()
        db.add_record(record_dict)

        return flask.jsonify(
            {
                "status": "saved",
                "message": f"Record {r_id} scored and saved (fraud probability: {r_pred}%)",
                "record_id": r_id,
                "fraud_probability": r_pred,
            }
        )

    except Exception as e:
        return (
            flask.jsonify(
                {"status": "error", "message": f"Error scoring record: {str(e)}"}
            ),
            500,
        )


@APP.route("/simulate", methods=["GET", "POST"])
def simulate():
    """Simulate scoring a random record from the simulation dataset.

    Randomly selects an event from simulate_data.json, scores it,
    and saves to database if new. Good for demos and testing.
    """
    try:
        # Load a random record from simulate data
        simulate_path = DATA_PROCESSED / "simulate_data.json"

        with open(simulate_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return (
                    flask.jsonify(
                        {"status": "error", "message": "No simulation data found"}
                    ),
                    404,
                )

            raw_record = json.loads(random.choice(lines))

        # Score the record
        r_id, r_pred, record = score_event_record(raw_record)

        # Check if already exists
        if db.record_exists(r_id):
            return flask.jsonify(
                {
                    "status": "exists",
                    "message": f"Record {r_id} already scored (fraud probability: {r_pred}%)",
                    "record_id": r_id,
                    "fraud_probability": r_pred,
                }
            )

        # Save new record to database
        record_dict = record.processed_record.to_dict()
        db.add_record(record_dict)

        return flask.jsonify(
            {
                "status": "success",
                "message": f"Simulated record {r_id} scored and saved!",
                "record_id": r_id,
                "fraud_probability": r_pred,
            }
        )

    except Exception as e:
        return (
            flask.jsonify(
                {"status": "error", "message": f"Error simulating record: {str(e)}"}
            ),
            500,
        )


if __name__ == "__main__":
    APP.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
