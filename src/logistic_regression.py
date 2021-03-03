import eli5
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
)

from a1_Data_Prep_Pipeline import load_fraud_data
from nlp_feature_engineering import quick_model_metrics


def setup_data(
    data,
    features,
    target_column="is_fraud",
    test_set_size=0.25,
    random_seed=7,
):
    """
    Validations features to ensure all are numeric,
    then performs a train/test split.

    Args:
        data (dataframe): Input data.
        features (list of strings): Features to use in modeling.
        target_column (str, optional): Defaults to "is_fraud".
        test_set_size (float, optional): Defaults to 0.25.
        random_seed (int, optional): For train/test split. Defaults to 7.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    null_values = data[features].isnull().sum().sum()
    if null_values > 0:
        print("The data has missing values...")
        print(data[features].isnull().sum(axis=0))
    non_numeric_features = 0
    for feature in features:
        if not is_numeric_dtype(data[feature]):
            feature_datatype = data[feature].dtype
            non_numeric_features += 1
            print(
                f"The feature> {feature} <is not numeric. Datatype: {feature_datatype}"
            )
    if (non_numeric_features != 0) or (null_values != 0):
        exit()
    X_train, X_test, y_train, y_test = train_test_split(
        data[features],
        data[target_column],
        test_size=test_set_size,
        random_state=random_seed,
    )
    return X_train, X_test, y_train, y_test


def create_pipeline(
    model,
    model_params={},
    scale_data=True,
    select_features=True,
    num_features_to_use="all",
):
    scaler = None
    feature_selector = None
    if scale_data:
        scaler = StandardScaler()
    if select_features:
        feature_selector = SelectKBest(k=num_features_to_use)
    model = model(**model_params)
    clf_pipeline = make_pipeline(scaler, feature_selector, model)
    print(clf_pipeline)
    return clf_pipeline


if __name__ == "__main__":
    # Load Data
    data = load_fraud_data("model_data_v2")
    # Select features to use for modeling.
    all_possible_features = [
        "approx_payout_date",
        "body_length",
        "channels",
        "country",
        "currency",
        "delivery_method",
        "description",
        "email_domain",
        "event_created",
        "event_end",
        "event_published",
        "event_start",
        "fb_published",
        "gts",
        "has_analytics",
        "has_header",
        "has_logo",
        "listed",
        "name",
        "name_length",
        "num_order",
        "num_payouts",
        "object_id",
        "org_desc",
        "org_facebook",
        "org_name",
        "org_twitter",
        "payee_name",
        "payout_type",
        "sale_duration",
        "sale_duration2",
        "show_map",
        "user_age",
        "user_created",
        "user_type",
        "venue_address",
        "venue_country",
        "venue_latitude",
        "venue_longitude",
        "venue_name",
        "venue_state",
        "is_fraud",
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
    top_features = []
    features_to_test = [
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
    # Create training and test datasets.
    X_train, X_test, y_train, y_test = setup_data(data, features_to_test)

    log_reg_best_params = dict(
        penalty="l2",
        C=1,
        class_weight=None,
        random_state=7,
        solver="newton-cg",
        max_iter=1000,
        tol=0.0001,
        l1_ratio=None,
    )

    clf_pipeline = create_pipeline(
        LogisticRegression,
        model_params=log_reg_best_params,
        scale_data=True,
        select_features=True,
        num_features_to_use=20,
    )

    clf_pipeline.fit(X_train, y_train)

    # log_reg_param_grid = dict(
    #     logisticregression__penalty=["l2"],
    #     logisticregression__C=[0.1, 0.5, 1, 5, 10, 100],
    #     logisticregression__class_weight=[None],
    #     logisticregression__random_state=[7],
    #     logisticregression__solver=[
    #         "newton-cg",
    #         "lbfgs",
    #         "sag",
    #         "saga",
    #         "liblinear",
    #     ],
    #     logisticregression__max_iter=[10, 100, 1000, 10000],
    #     logisticregression__tol=[0.001, 0.0001],
    #     logisticregression__l1_ratio=[None],
    # )

    # grid_search_cv_pipeline = GridSearchCV(
    #     clf_pipeline, log_reg_param_grid, verbose=2
    # )

    # grid_search_cv_pipeline.fit(X_train, y_train)

    # print(grid_search_cv_pipeline.best_params_)
    # print(grid_search_cv_pipeline.best_estimator_)
    # print(grid_search_cv_pipeline.best_score_)

    quick_model_metrics(
        clf_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )