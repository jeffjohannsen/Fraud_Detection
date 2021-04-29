import flask
import os
import sys

# Need to update sys path to import below modules from parent directory.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a3_Live_Record_Processing import EventRecordManager, EventRecord
from confidential import (
    server,
    fraud_detection_db_1_endpoint,
    fraud_detection_db_1_password,
)


# Function for queries to RDS database.
def query_records_table():
    record_count_query = """
                            SELECT COUNT(1)
                            FROM fraud_records_1
                         """
    high_count_query = """
                        SELECT COUNT(1)
                        FROM fraud_records_1
                        WHERE fraud_proba >= 0.5
                       """
    med_count_query = """
                        SELECT COUNT(1)
                        FROM fraud_records_1
                        WHERE fraud_proba >= 0.1 AND fraud_proba < 0.5
                      """
    low_count_query = """
                        SELECT COUNT(1)
                        FROM fraud_records_1
                        WHERE fraud_proba < 0.1
                      """
    records_query = """
                    SELECT object_id,
                        event_created,
                        country,
                        CAST(avg_ticket_cost AS MONEY),
                        CAST(total_ticket_value AS MONEY),
                        fraud_proba,
                        CASE
                            WHEN fraud_proba >= 0.5 THEN 'High'
                            WHEN fraud_proba >= 0.1 AND fraud_proba < 0.5 THEN 'Medium'
                            ELSE 'Low'
                        END AS fraud_risk_level
                        FROM fraud_records_1
                        ORDER BY fraud_proba DESC, country
                    """
    record_count = 0
    records = None
    with manager.rds_engine.connect() as connection:
        record_count = [x[0] for x in connection.execute(record_count_query)][
            0
        ]
        high_count = [x[0] for x in connection.execute(high_count_query)][0]
        med_count = [x[0] for x in connection.execute(med_count_query)][0]
        low_count = [x[0] for x in connection.execute(low_count_query)][0]
        high_perc = round((high_count / record_count) * 100, 1)
        med_perc = round((med_count / record_count) * 100, 1)
        low_perc = round((low_count / record_count) * 100, 1)
        records = connection.execute(records_query).fetchall()
    return {
        "record_count": record_count,
        "high_count": high_count,
        "med_count": med_count,
        "low_count": low_count,
        "high_perc": high_perc,
        "med_perc": med_perc,
        "low_perc": low_perc,
        "records": records,
    }


# Flask App Setup
APP = flask.Flask(__name__)


@APP.route("/")
def home():
    data = query_records_table()
    return flask.render_template("home.html", **data)


@APP.route("/dashboard")
def dashboard():
    return flask.render_template("dashboard.html")


@APP.route("/score", methods=["POST"])
def score():
    raw_record = flask.request.get_json()
    record = EventRecord()
    record.load_record_from_json(raw_record)
    record.clean_record()
    record.convert_html_to_text()
    record.create_nlp_predictions(
        {
            "name": manager.nlp_model_name,
            "description": manager.nlp_model_description,
            "org_name": manager.nlp_model_org_name,
            "org_desc": manager.nlp_model_org_desc,
        }
    )
    record.predict_record(manager.final_model)
    r_id = record.record_id
    r_pred = round(record.final_prediction * 100, 1)
    if record.record_id in manager.already_predicted:
        return f"Record {r_id} with a fraud probability of {r_pred}% already in database."
    else:
        record.save_record(manager.rds_engine, "fraud_records_1")
        manager.already_predicted.add(record.record_id)
        return f"Record {r_id} with a fraud probability of {r_pred}% saved."


if __name__ == "__main__":
    model_dict = {
        "nlp_model_description": "nlp_description_text_clf_pipeline",
        "nlp_model_name": "nlp_name_text_clf_pipeline",
        "nlp_model_org_desc": "nlp_org_desc_text_clf_pipeline",
        "nlp_model_org_name": "nlp_org_name_text_clf_pipeline",
        "final_model": "forest_clf",
    }
    manager = EventRecordManager(
        server=server,
        model_dict=model_dict,
        rds_endpoint=fraud_detection_db_1_endpoint,
        rds_password=fraud_detection_db_1_password,
    )
    APP.debug = False
    APP.run(host="0.0.0.0", port=8000)
