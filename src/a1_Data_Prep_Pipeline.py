import pandas as pd
from bs4 import BeautifulSoup


def load_fraud_data(filename):
    """
    Loads fraud detection data from specifies file.

    Args:
        filename (string): Data filename without .json

    Returns:
        dataframe
    """
    filepath = f"../data/{filename}.json"
    df = pd.read_json(filepath, orient="records", lines=True)
    return df


def create_target(df):
    """
    Determines fraud or not fraud from acct_type
    and removes acct_type.

    Args:
        df (Dataframe)

    Returns:
        Dataframe
    """
    df["is_fraud"] = df["acct_type"].apply(
        lambda x: "Fraud" if "fraud" in x else "Not Fraud"
    )
    df = df.drop(columns=["acct_type"])
    return df


def fill_missing_values(df):
    """
    Fills missing values and replaces empty string values with "Unknown".

    Args:
        df (Dataframe)

    Returns:
        Dataframe
    """
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
        df[feature] = df[feature].replace("", "Unknown")

    null_string_features = [
        "country",
        "venue_country",
        "venue_name",
        "venue_state",
    ]
    for feature in null_string_features:
        df[feature] = df[feature].fillna("Unknown")

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
        df[feature] = df[feature].fillna(-1)

    return df


def convert_datetimes(df):
    """
    Converts datetime features to proper format.
    Args:
        df (Dataframe)

    Returns:
        Dataframe
    """
    datetime_features = [
        "approx_payout_date",
        "event_created",
        "event_end",
        "event_published",
        "event_start",
        "user_created",
    ]
    for feature in datetime_features:
        df[feature] = pd.to_datetime(df[feature], unit="s")
    return df


def aggregate_nested_features(df):
    """
    Replaces nested list and dictionary features with new aggregate features.

    Args:
        df (dataframe)

    Returns:
        dataframe
    """
    # Helper functions for applying aggregations.
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
    df["num_previous_payouts"] = df["previous_payouts"].apply(lambda x: len(x))
    df["previous_payouts_total"] = df["previous_payouts"].apply(payouts_sum)
    df = df.drop(columns=["previous_payouts"])
    # Ticket Types
    df["num_ticket_types"] = df["ticket_types"].apply(lambda x: len(x))
    df["num_tickets_available"] = df["ticket_types"].apply(
        get_tickets_available
    )
    df["total_ticket_value"] = df["ticket_types"].apply(get_total_ticket_value)
    df["avg_ticket_cost"] = (
        df["total_ticket_value"] / df["num_tickets_available"]
    )
    df["avg_ticket_cost"] = df["avg_ticket_cost"].fillna(0)
    df = df.drop(columns=["ticket_types"])

    return df


def convert_categorical_features(df):
    """
    Replaces certain categorical features with a binary value.

    Args:
        df (dataframe)

    Returns:
        dataframe
    """
    df["known_payee_name"] = df["payee_name"].apply(
        lambda x: 0 if x == "Unknown" else 1
    )
    df["known_venue_name"] = df["venue_name"].apply(
        lambda x: 0 if x == "Unknown" else 1
    )
    df["known_payout_type"] = df["payout_type"].apply(
        lambda x: 0 if x == "Unknown" else 1
    )
    return df


def remove_features(df):
    """
    Removing features that are not to be used in any modeling.

    Args:
        df (dataframe)

    Returns:
        dataframe
    """
    data_leakage_features = []
    non_useful_features = []
    df = df.drop(columns=data_leakage_features + non_useful_features)
    return df


def convert_html_to_text(df, html_features):
    """
    Converts html features to plain text.

    Args:
        df (dataframe)
        html_features (list of strings): Names of html features.

    Returns:
        dataframe
    """
    for feature in html_features:
        df[feature] = df[feature].apply(
            lambda x: BeautifulSoup(x, "html.parser")
        )
        df[feature] = df[feature].apply(lambda x: x.get_text("|", strip=True))
    return df


def run_data_prep_pipeline(original_df):
    """
    Runs all steps in the data cleaning, data preparation,
    and feature engineering process.

    Args:
        original_df (dataframe): Unedited original data.

    Returns:
        dataframe: Model ready data.
    """
    df = original_df.copy()
    df = create_target(df)
    df["listed"] = df["listed"].replace({"y": 1, "n": 0})
    df = convert_datetimes(df)
    df = fill_missing_values(df)

    df = aggregate_nested_features(df)
    df = convert_categorical_features(df)
    df["total_empty_values"] = df.applymap(
        lambda x: x in ["Unknown", -1, 0]
    ).sum(axis=1)
    df = convert_html_to_text(df, ["description", "org_desc"])
    df = remove_features(df)
    return df


def save_df_to_json(df, filename):
    """
    Saves data to json file in data folder.

    Args:
        df (dataframe): Data to save.
        filename (string): Name of file to create and save to.
    """
    filepath = f"../data/{filename}.json"
    df.to_json(filepath, orient="records", lines=True)
    print(f"Data saved to {filename}.json in the data folder.")


if __name__ == "__main__":
    original_working_data = load_fraud_data("working_data")
    model_ready_data = run_data_prep_pipeline(original_working_data)
    # print(model_ready_data.info())
    save_df_to_json(model_ready_data, "model_data_v1")