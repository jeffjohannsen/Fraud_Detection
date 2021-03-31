import flask

from ..a3_Live_Record_Processing import EventRecordManager
from ..confidential import (
    server,
    fraud_detection_db_1_endpoint,
    fraud_detection_db_1_password,
)

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


APP = flask.Flask(__name__)


@APP.route("/")
def index():
    return flask.render_template("index.html")


@APP.route("/score")
def score():
    pass


@APP.route("/get_text_prediction")
def get_text_prediction():
    pass


@APP.route("/dashboard")
def make_dashboard():
    pass


if __name__ == "__main__":
    APP.debug = True
    APP.run()