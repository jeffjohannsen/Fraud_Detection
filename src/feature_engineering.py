import pandas as pd

from data_cleaning import load_data_fd


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


def aggregate_nested_features(data):
    data["num_previous_payouts"] = data["previous_payouts"].apply(
        lambda x: len(x)
    )
    data["previous_payouts_total"] = data["previous_payouts"].apply(
        payouts_sum
    )

    data = data.drop(columns=["previous_payouts"])

    data["num_ticket_types"] = data["ticket_types"].apply(lambda x: len(x))
    data["num_tickets_available"] = data["ticket_types"].apply(
        get_tickets_available
    )
    data["total_ticket_value"] = data["ticket_types"].apply(
        get_total_ticket_value
    )
    data["avg_ticket_cost"] = (
        data["total_ticket_value"] / data["num_tickets_available"]
    )

    data["avg_ticket_cost"] = data["avg_ticket_cost"].fillna(-1)

    data = data.drop(columns=["ticket_types"])
    return data


def replace_datetime_features(data):
    datetime_features = [
        "approx_payout_date",
        "event_created",
        "event_end",
        "event_start",
        "user_created",
    ]
    for feature in datetime_features:
        data[feature] = pd.to_datetime(data[feature], unit="ms")

    data["days_from_event_created_till_start"] = (
        data["event_start"] - data["event_created"]
    ).dt.days

    datetime_features = [
        "approx_payout_date",
        "event_created",
        "event_end",
        "event_start",
        "user_created",
    ]
    data = data.drop(columns=datetime_features)
    return data


def convert_categorical_features(data):
    data["unknown_payee_name"] = data["payee_name"].apply(
        lambda x: 1 if x == "Unknown" else 0
    )
    data["unknown_venue_name"] = data["venue_name"].apply(
        lambda x: 1 if x == "Unknown" else 0
    )
    data = pd.get_dummies(data, columns=["payout_type"], drop_first=True)
    return data


if __name__ == "__main__":

    data = load_data_fd("clean_data")[1]
    data = aggregate_nested_features(data)
    data = replace_datetime_features(data)
    data = convert_categorical_features(data)

    # print(data.info())

    data.to_json("../data/model_data_v1.json", orient="records", lines=True)
