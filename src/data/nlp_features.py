import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data.preprocessing import load_fraud_data, save_df_to_json

pd.set_option("display.max_columns", 100)


def separate_text_features_target(df):
    """
    Creates a new dataframe with only text features:
    Description, Name, Organization Description, Organization Name

    Args:
        df (dataframe): Full dataframe

    Returns:
        dataframe: Only text feature dataframe
    """
    text_columns = ["description", "name", "org_desc", "org_name", "is_fraud"]
    df_text = df.loc[:, text_columns].copy()
    return df_text


def quick_model_metrics(fitted_model, X_train, y_train, X_test, y_test):
    """
    Model accuracy vs baseline (majority class - Not Fraud)
    for training and testing datasets.

    Args:
        fitted_model (object)
        X_train (dataframe)
        y_train (dataframe)
        X_test (series)
        y_test (series)
    """
    train_baseline = y_train[y_train == "Not Fraud"].size / y_train.size
    test_baseline = y_test[y_test == "Not Fraud"].size / y_test.size
    train_accuracy = fitted_model.score(X_train, y_train)
    test_accuracy = fitted_model.score(X_test, y_test)
    print(f"Train Baseline: {train_baseline:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Baseline: {test_baseline:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")


def save_fitted_model(model, filename):
    """
    Save fitted model to models folder in pickle format.

    Args:
        model (object): Fitted model
        filename (string): Model name
    """
    from src.config import MODELS_DIR

    filepath = MODELS_DIR / f"{filename}.pkl"
    with open(filepath, "wb") as file:
        pkl.dump(model, file)
    print("Model saved successfully.")


def convert_text_to_predict_proba(df, features_to_convert):
    """
    Runs a saved model pipeline on text features.
    Adds prediction probability feature to dataframe
    for each text feature.

    Args:
        df (dataframe)
        features_to_convert (list of strings)

    Returns:
        dataframe
    """
    from src.config import MODELS_DIR

    modeled_features = ["name", "description", "org_name", "org_desc"]
    for feature in features_to_convert:
        if feature in modeled_features:
            model = None
            with open(MODELS_DIR / f"nlp_{feature}_text_clf_pipeline.pkl", "rb") as f:
                model = pkl.load(f)

            fraud_index = np.argwhere(model.classes_ == "Fraud")[0, 0]
            predictions = model.predict_proba(df[feature])[:, fraud_index]
            df[f"{feature}_proba"] = predictions
        else:
            print(f"Error: {feature} is not available to be converted.")
            print("Current possible features include: ")
            print(modeled_features)
    return df


def text_to_predict_proba(X_train, y_train, X_test, features_to_convert):
    """
    Converts text features to predict_proba of fraud.
    Trains the model each time to specific inputted
    train test split to avoid any possible data leakage.
    Time consuming at scale.

    Args:
        X_train (dataframe)
        y_train (series)
        X_test (dataframe)
        features_to_convert (list of strings):
            List of text feature names to model
            and convert to probability of fraud.

    Returns:
        tuple of dataframes: X_train, X_test
    """
    train = X_train
    test = X_test
    for feature in features_to_convert:
        text_clf_pipeline = Pipeline(
            [
                (
                    "vect",
                    CountVectorizer(ngram_range=(1, 2), stop_words="english"),
                ),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    SGDClassifier(loss="modified_huber", max_iter=10000),
                ),
            ]
        )
        text_clf_pipeline.fit(train[feature], y_train)
        fraud_index = np.argwhere(text_clf_pipeline.classes_ == "Fraud")[0, 0]
        for dataset in [train, test]:
            predictions = text_clf_pipeline.predict_proba(dataset[feature])[
                :, fraud_index
            ]
            dataset[f"{feature}_proba"] = predictions
            dataset.drop(columns=feature, inplace=True)
    return train, test


if __name__ == "__main__":
    # Loading in data
    df = load_fraud_data("model_data_v1")

    text_df = separate_text_features_target(df)
    # Choose text feature to process
    current_feature = "description"
    # Make train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        text_df[current_feature],
        text_df["is_fraud"],
        test_size=0.25,
        random_state=10,
    )
    # Create text processing system
    text_clf_pipeline = Pipeline(
        [
            (
                "vect",
                CountVectorizer(ngram_range=(1, 2), stop_words="english"),
            ),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                SGDClassifier(loss="modified_huber", max_iter=10000),
            ),
        ]
    )
    # Train text processing system
    text_clf_pipeline.fit(X_train, y_train)
    # Check performance of text processing system
    quick_model_metrics(
        text_clf_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    # # Save model for predictions to be done later
    # save_fitted_model(
    #     text_clf_pipeline, f"nlp_{current_feature}_text_clf_pipeline"
    # )

    # # Load model and predict. Adding predictions to new dataset.
    # df = convert_text_to_predict_proba(
    #     df, ["name", "description", "org_name", "org_desc"]
    # )
    # save_df_to_json(df, "model_data_v2")
