import flask

# from home.jeff.Documents.Data_Science_Projects.Fraud_Detection.src.a3_Live_Record_Processing import (
#     EventRecordManager,
# )
# from src.confidential import (
#     server,
#     fraud_detection_db_1_endpoint,
#     fraud_detection_db_1_password,
# )

# model_dict = {
#     "nlp_model_description": "nlp_description_text_clf_pipeline",
#     "nlp_model_name": "nlp_name_text_clf_pipeline",
#     "nlp_model_org_desc": "nlp_org_desc_text_clf_pipeline",
#     "nlp_model_org_name": "nlp_org_name_text_clf_pipeline",
#     "final_model": "forest_clf",
# }
# manager = EventRecordManager(
#     server=server,
#     model_dict=model_dict,
#     rds_endpoint=fraud_detection_db_1_endpoint,
#     rds_password=fraud_detection_db_1_password,
# )


APP = flask.Flask(__name__)


@APP.route("/")
def home():
    return flask.render_template("home.html")


@APP.route("/dashboard")
def dashboard():
    return flask.render_template("dashboard.html")


@APP.route("/score")
def score():
    pass


if __name__ == "__main__":
    APP.debug = True
    APP.run()
