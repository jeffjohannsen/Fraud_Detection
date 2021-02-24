import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data_cleaning import load_data_fd

if __name__ == "__main__":
    data = load_data_fd("model_data_v1")[1]

    target = data["is_fraud"]
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
        "object_id",
        "org_facebook",
        "org_twitter",
        "sale_duration",
        "sale_duration2",
        "show_map",
        "user_age",
        "user_type",
        "venue_latitude",
        "venue_longitude",
        "num_previous_payouts",
        "previous_payouts_total",
        "num_ticket_types",
        "num_tickets_available",
        "total_ticket_value",
        "avg_ticket_cost",
        "days_from_event_created_till_start",
        "unknown_payee_name",
        "unknown_venue_name",
        "payout_type_CHECK",
        "payout_type_Unknown",
    ]
    features = data[features_to_use]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=7
    )

    log_reg_clf = LogisticRegression(
        penalty="l2",
        C=1,
        class_weight=None,
        random_state=7,
        solver="newton-cg",
        max_iter=1000,
        tol=0.0001,
        l1_ratio=None,
    )

    log_reg_clf.fit(X_train, y_train)

    train_baseline = y_train[y_train == False].size / y_train.size
    test_baseline = y_test[y_test == False].size / y_test.size

    train_accuracy = log_reg_clf.score(X_train, y_train)
    test_accuracy = log_reg_clf.score(X_test, y_test)

    print(f"Train Baseline: {train_baseline:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")

    print(f"Test Baseline: {test_baseline:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
