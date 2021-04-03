import dash
import dash_table
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
from sqlalchemy import create_engine

from confidential import (
    fraud_detection_db_1_password,
    fraud_detection_db_1_endpoint,
)

# Connecting to the database
db_pass = fraud_detection_db_1_password
db_loc = fraud_detection_db_1_endpoint
rds_engine = create_engine(
    f"postgresql+psycopg2://postgres:{db_pass}@{db_loc}/fraud_detection_db_1"
)
query = """
        SELECT object_id,
        fraud_proba,
        record_predicted_datetime,
        event_created,
        venue_latitude,
        venue_longitude,
        num_tickets_available,
        avg_ticket_cost,
        total_ticket_value,
        name_proba,
        description_proba,
        org_name_proba,
        org_desc_proba,
        CASE
            WHEN fraud_proba > 0.5 THEN 'High'
            WHEN fraud_proba > 0.1 THEN 'Medium'
            ELSE 'Low'
        END AS severity_level
        FROM fraud_records_1
        ;
"""
df = None
with rds_engine.connect() as connection:
    df = pd.read_sql(sql=query, con=connection)

# Dash web app dashboard
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.Div([html.H1("Overview")], style={"float": "left"}),
        html.Div(
            [
                html.H2("Event Records"),
                dash_table.DataTable(
                    id="fraud_record_table",
                    columns=[
                        {"name": "Event ID", "id": "object_id"},
                        {
                            "name": "Datetime",
                            "id": "record_predicted_datetime",
                        },
                        {"name": "Fraud Probability", "id": "fraud_proba"},
                        {"name": "Total Exposure", "id": "total_ticket_value"},
                        {"name": "Severity", "id": "severity_level"},
                    ],
                    data=df.to_dict("records"),
                    sort_action="native",
                ),
            ],
            style={"float": "right"},
        ),
    ],
    style={},
)


if __name__ == "__main__":
    app.run_server(debug=True)
