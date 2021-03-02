import pandas as pd
import pickle as pkl
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from a1_Data_Prep_Pipeline import load_fraud_data


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
    filepath = f"../models/{filename}.pkl"
    with open(filepath, "wb") as file:
        pkl.dump(model, file)
    print("Model saved successfully.")


if __name__ == "__main__":
    # Loading in data
    df = load_fraud_data("model_data_v1")
    text_df = separate_text_features_target(df)
    # Choose text feature to process
    current_feature = "org_name"
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
    # Save model for predictions to be done later
    # save_fitted_model(
    #     text_clf_pipeline, f"nlp_{current_feature}_text_clf_pipeline"
    # )
