import csv
import time
import requests
import pandas as pd
import numpy as np
import pickle as pkl
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from datetime import datetime
from bs4 import BeautifulSoup

from confidential import (
    server,
    fraud_detection_db_1_password,
    fraud_detection_db_1_endpoint,
)
from a1_Data_Prep_Pipeline import convert_datetimes


class EventRecord:
    def __init__(self):
        self.record_id = None
        self.record_datetime = None
        self.record_predicted_datetime = None
        self.original_record = None
        self.processed_record = None
        self.nlp_predictions = {
            "name": None,
            "description": None,
            "org_name": None,
            "org_desc": None,
        }
        self.final_prediction = None

    def load_record_from_json(self, record_json):
        self.record_id = record_json["object_id"]
        self.original_record = record_json
        self.record_predicted_datetime = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.record_datetime = datetime.fromtimestamp(
            record_json["event_created"]
        ).strftime("%Y/%m/%d %H:%M:%S")

    def load_and_verify_record(self, already_predicted, server):
        r = requests.get(server)
        r.raise_for_status()
        record_json = r.json()
        self.record_id = record_json["object_id"]
        if self.record_id in already_predicted:
            return True
        else:
            self.original_record = record_json
            self.record_predicted_datetime = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self.record_datetime = datetime.fromtimestamp(
                record_json["event_created"]
            ).strftime("%Y/%m/%d %H:%M:%S")

    def clean_record(self):
        record = pd.Series(self.original_record)
        record = convert_datetimes(record)
        # Dealing with Nan/Null Values
        empty_string_features = [
            "payee_name",
            "country",
            "description",
            "name",
            "org_desc",
            "org_name",
            "payout_type",
            "venue_country",
            "venue_name",
            "venue_state",
        ]
        for feature in empty_string_features:
            record[feature] = (
                "Unknown" if record[feature] == "" else record[feature]
            )
        null_string_features = [
            "country",
            "venue_country",
            "venue_name",
            "venue_state",
        ]
        for feature in null_string_features:
            record[feature] = (
                "Unknown" if pd.isnull(record[feature]) else record[feature]
            )
        null_num_features = [
            "delivery_method",
            "has_header",
            "org_facebook",
            "org_twitter",
            "sale_duration",
            "venue_latitude",
            "venue_longitude",
        ]
        for feature in null_num_features:
            record[feature] = (
                -1 if pd.isnull(record[feature]) else record[feature]
            )

        # Creating Aggregate Features
        def payouts_sum(x):
            total = 0
            for payout in x:
                total += payout["amount"]
            return total

        def get_tickets_available(x):
            total = 0
            for ticket_type in x:
                total += ticket_type["quantity_total"]
            return total

        def get_total_ticket_value(x):
            total = 0
            for ticket_type in x:
                total += ticket_type["quantity_total"] * ticket_type["cost"]
            return total

        # Previous Payouts
        record["num_previous_payouts"] = len(record["previous_payouts"])
        record["previous_payouts_total"] = payouts_sum(
            record["previous_payouts"]
        )
        # Ticket Types
        record["num_ticket_types"] = len(record["ticket_types"])
        record["num_tickets_available"] = get_tickets_available(
            record["ticket_types"]
        )
        record["total_ticket_value"] = get_total_ticket_value(
            record["ticket_types"]
        )
        # Ticket Costs
        if record["num_tickets_available"] == 0:
            record["avg_ticket_cost"] = 0
        else:
            record["avg_ticket_cost"] = (
                record["total_ticket_value"] / record["num_tickets_available"]
            )
        # Converting Categorical Features
        listed_dict = {"y": 1, "n": 0}
        record["listed"] = listed_dict[record["listed"]]
        record["known_payee_name"] = (
            0 if record["payee_name"] == "Unknown" else 1
        )
        record["known_venue_name"] = (
            0 if record["venue_name"] == "Unknown" else 1
        )
        record["known_payout_type"] = (
            0 if record["payout_type"] == "Unknown" else 1
        )

        record["total_empty_values"] = sum(
            [1 if x in ["Unknown", -1, 0] else 0 for x in record]
        )

        self.processed_record = record

    def add_features():
        pass

    def convert_html_to_text(self):
        record = self.processed_record
        for feature in ["description", "org_desc"]:
            record[feature] = BeautifulSoup(record[feature], "html.parser")
            record[feature] = record[feature].get_text("|", strip=True)
        self.processed_record = record

    def create_nlp_predictions(self, trained_nlp_models):
        for feature, model in trained_nlp_models.items():
            fraud_index = int(np.argwhere(model.classes_ == "Fraud"))
            prediction = model.predict_proba([self.processed_record[feature]])[
                :, fraud_index
            ]
            prediction = round(float(prediction), 5)
            self.processed_record[f"{feature}_proba"] = prediction
            self.nlp_predictions[feature] = prediction

    def predict_record(self, trained_clf_model):
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
        clf = trained_clf_model
        fraud_index = int(np.argwhere(clf.classes_ == "Fraud"))
        record_to_predict = (
            self.processed_record[features].to_numpy().reshape(1, -1)
        )
        prediction = clf.predict_proba(record_to_predict)[:, fraud_index]
        prediction = round(float(prediction), 5)
        self.final_prediction = prediction
        self.processed_record["fraud_proba"] = prediction
        self.processed_record[
            "record_predicted_datetime"
        ] = self.record_predicted_datetime

    def save_record(self, engine, table_name):
        with engine.connect() as connection:
            temp = pd.DataFrame(self.processed_record).T
            temp.to_sql(
                name=table_name,
                con=connection,
                if_exists="append",
                index=False,
                dtype={
                    "previous_payouts": postgresql.JSONB,
                    "ticket_types": postgresql.JSONB,
                },
            )


class EventRecordManager:
    """
    Keeps track of RDS database, ML models, and overall flow of records.
    """

    def __init__(self, server, model_dict, rds_endpoint, rds_password):
        """
        Initializes EventRecordManager. Manager keeps track
        of database, ML models, and overall flow of records.

        Args:
            server (str): Location of server providing records.
            model_dict (dict): Keys are attribute names
                and values are filenames containing trained models.
                Current models required:
                    nlp_model_description: filename,
                    nlp_model_name: filename,
                    nlp_model_org_desc: filename,
                    nlp_model_org_name: filename,
                    final_model: filename
                NLP models are text transform pipelines with a clf at the end.
                Final model is a clf.
            rds_endpoint (str): Database location. Shown on AWS RDS console.
            rds_password (str): Database password.
        """
        self.model_dict = {
            "nlp_model_description": "nlp_description_text_clf_pipeline",
            "nlp_model_name": "nlp_name_text_clf_pipeline",
            "nlp_model_org_desc": "nlp_org_desc_text_clf_pipeline",
            "nlp_model_org_name": "nlp_org_name_text_clf_pipeline",
            "final_model": "forest_clf",
        }
        self.already_predicted = set()
        self.final_model = None
        self.nlp_model_description = None
        self.nlp_model_name = None
        self.nlp_model_org_desc = None
        self.nlp_model_org_name = None
        self.records_pulled = 0
        self.records_predicted_and_saved = 0
        self.server = server
        self.rds_engine = None
        self.load_models(model_dict)
        self.connect_to_aws_rds(rds_endpoint, rds_password)
        self.load_already_predicted()

    # TODO: Security issues with pickle.
    # TODO: Verify valid model_name.
    # TODO: Verify valid model.
    def load_model(
        self,
        model_name,
        filename,
    ):
        """
        Add saved model to attributes.

        Args:
            model_name (str): attribute name
            filename (str): filename where model is stored.
        """
        with open(f"../../models/{filename}.pkl", "rb") as file:
            model = pkl.load(file)
            setattr(self, model_name, model)

    # TODO: Verify valid model_names.
    def load_models(self, models_to_load):
        """
        Loads multiple models to attributes.

        Args:
            models_to_load (dict): Keys are attribute names
                and values are filenames.
        """
        for k, v in models_to_load.items():
            self.load_model(k, v)

    def load_already_predicted(self):
        """
        Loads set of already predicted records from database.
        """
        pred_list = []
        with self.rds_engine.connect() as connection:
            result = connection.execute(
                "SELECT object_id from fraud_records_1"
            )
            pred_list = [x[0] for x in result]
        self.already_predicted = set(pred_list)

    def connect_to_aws_rds(self, endpoint, password):
        """
        Connects to the PostgreSQL database on AWS RDS.

        Args:
            endpoint (str): Database location. Shown on AWS RDS console.
            password (str): Database password.
        """
        self.rds_engine = create_engine(
            f"postgresql+psycopg2://postgres:{password}@{endpoint}/fraud_detection_db_1"
        )

    def process_records_local(self, iter, sleep, print_results=True):
        """
        Runs the record processing pipeline for specified iterations.
        Pulls records from server, processes, predicts, and saves to AWS RDS.

        Args:
            iter (int): Number of records to pull and process.
            sleep (int): Seconds to wait in between calls to server.
            print_results (bool, optional): Whether to print results of each record.
                                            Defaults to True.
        """
        for _ in range(iter):
            time.sleep(sleep)
            record = EventRecord()
            verification = record.load_and_verify_record(
                self.already_predicted, self.server
            )
            self.records_pulled += 1
            if verification:
                if print_results:
                    print(".")
                continue
            else:
                self.already_predicted.add(record.record_id)
                record.clean_record()
                record.convert_html_to_text()
                record.create_nlp_predictions(
                    {
                        "name": self.nlp_model_name,
                        "description": self.nlp_model_description,
                        "org_name": self.nlp_model_org_name,
                        "org_desc": self.nlp_model_org_desc,
                    }
                )
                record.predict_record(self.final_model)
                record.save_record(self.rds_engine, "fraud_records_1")
                self.records_predicted_and_saved += 1
                if print_results:
                    r_id = record.record_id
                    r_pred = round(record.final_prediction * 100, 1)
                    print(
                        f"Record {r_id} with a fraud probability of {r_pred}% saved."
                    )
        if print_results:
            rp = self.records_pulled
            rps = self.records_predicted_and_saved
            print(f"Records Pulled: {rp}, Records Predicted and Saved: {rps}")
        self.records_pulled = 0
        self.records_predicted_and_saved = 0

    def process_records_aws(self, iter, sleep, print_results=True):
        """
        Pulls record from server and posts JSON to Flask App /score endpoint
        which runs the processing pipeline.

        Pulls records from server, processes, predicts, and saves to AWS RDS.

        Args:
            iter (int): Number of records to pull and process.
            sleep (int): Seconds to wait in between calls to server.
            print_results (bool, optional): Whether to print results of each record.
                                            Defaults to True.
        """
        for _ in range(iter):
            server_record = requests.get(server)
            record_json = server_record.json()
            aws_url = "ec2-34-223-178-205.us-west-2.compute.amazonaws.com"
            requests.post(aws_url, json=record_json)
            time.sleep(sleep)


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

    manager.process_records_aws(10, 2)
