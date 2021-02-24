import pandas as pd


def create_target(data):
    """
    Determines fraud or not fraud from acct_type
    and removes acct_type.

    Args:
        data (Dataframe)

    Returns:
        Dataframe
    """
    data["is_fraud"] = data["acct_type"].apply(
        lambda x: True if "fraud" in x else False
    )
    data = data.drop(columns=["acct_type"])
    return data


def convert_datatypes(data):
    """
    Args:
        data (Dataframe)

    Returns:
        Dataframe
    """
    data["listed"] = data["listed"].replace({"y": 1, "n": 0})
    datetime_features = [
        "approx_payout_date",
        "event_created",
        "event_end",
        "event_published",
        "event_start",
        "user_created",
    ]
    for feature in datetime_features:
        data[feature] = pd.to_datetime(data[feature], unit="s")
    return data


def fill_nan_nulls(data):
    """
    Args:
        data (Dataframe)

    Returns:
        Dataframe
    """
    replace_empty_strings = [
        "payee_name",
        "country",
        "description",
        "name",
        "org_desc",
        "payout_type",
        "venue_country",
        "venue_name",
        "venue_state",
    ]
    for feature in replace_empty_strings:
        data[feature] = data[feature].replace("", "Unknown")

    fill_null_string = [
        "country",
        "venue_country",
        "venue_name",
        "venue_state",
    ]
    for feature in fill_null_string:
        data[feature] = data[feature].fillna("Unknown")

    fill_null_num = [
        "delivery_method",
        "has_header",
        "org_facebook",
        "org_twitter",
        "sale_duration",
        "venue_latitude",
        "venue_longitude",
    ]
    for feature in fill_null_num:
        data[feature] = data[feature].fillna(-1)
    return data


def drop_unused_features(data):
    """
    Args:
        data (Dataframe)

    Returns:
        Dataframe
    """
    data = data.drop(columns=["event_published", "venue_address"])
    return data


def load_data_fd(filename):
    """
    Loads fraud detection data from specifies file.

    Args:
        filename (string): Data filename without .json

    Returns:
        tuple of dataframes: original_data and copy of original data
    """
    filepath = f"../data/{filename}.json"
    original_data = pd.read_json(filepath, orient="records", lines=True)
    data = original_data.copy()
    return original_data, data


if __name__ == "__main__":
    # Load in working data
    data = load_data_fd("working_data")[1]

    # Clean data
    data = create_target(data)
    data = convert_datatypes(data)
    data = fill_nan_nulls(data)
    data = drop_unused_features(data)

    # Save to cleaned data file
    data.to_json("../data/clean_data.json", orient="records", lines=True)
