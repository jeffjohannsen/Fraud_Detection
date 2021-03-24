import pickle as pkl
import requests
import time
import pprint
import pandas as pd
import numpy as np

from a1_Data_Prep_Pipeline import (
    aggregate_nested_features,
    convert_datetimes,
    fill_missing_values,
    convert_categorical_features,
    convert_html_to_text,
    remove_features,
)

from nlp_feature_engineering import convert_text_to_predict_proba


def get_event(already_predicted):
    """
    Pulls a record from the server
    and verifies that it is not
    a recently pulled record.

    Args:
        already_predicted (list of ints): List of id numbers of already
                                          pulled records.

    Returns:
        None: If the record is already pulled.
        OR
        Tuple: JSON string and updated already predicted list.
    """
    server_response = requests.get(
        "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"
    )
    response = server_response.json()
    id = response["object_id"]
    if id in already_predicted:
        return None
    else:
        already_predicted.append(id)
        return response, already_predicted


def convert_record(record):
    """
    Converts raw Json record into model ready data.
    Includes data processing pipeline and
    text NLP modeling.

    Args:
        record (str): Json string of event record.

    Returns:
        Dataframe: Processed record.
    """
    record = pd.DataFrame({k: [v] for k, v in record.items()})
    record["listed"] = record["listed"].replace({"y": 1, "n": 0})
    record = convert_datetimes(record)
    record = fill_missing_values(record)

    record = aggregate_nested_features(record)
    record = convert_categorical_features(record)
    record["total_empty_values"] = record.applymap(lambda x: x in [0, -1]).sum(
        axis=1
    )
    record = convert_html_to_text(record, ["description", "org_desc"])
    record = remove_features(record)
    record = convert_text_to_predict_proba(
        record, ["name", "description", "org_name", "org_desc"]
    )
    features_to_use = [
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
    record = record[features_to_use]
    return record


def predict_record(record, clf_name):
    """
    Runs model on event record to predict
    fraudulent or not.

    Args:
        record (Dataframe): Processes event record.
        clf (str): Name of model to use for classification.
    """
    clf = None
    with open(f"../models/{clf_name}.pkl", "rb") as f:
        clf = pkl.load(f)
    fraud_index = int(np.argwhere(clf.classes_ == "Fraud"))
    fraud_proba = float(clf.predict_proba(record)[:, fraud_index])
    record["fraud_proba"] = fraud_proba
    return record


predicted_list = []

for i in range(1000):
    time.sleep(2)
    event = get_event(predicted_list)
    if event is None:
        print(".")
        continue
    else:
        predicted_list = event[1]
        event = convert_record(event[0])
        event = predict_record(event, "forest_clf")
        print(event.info())

# print(predicted_list)
