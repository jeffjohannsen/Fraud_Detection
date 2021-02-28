import pandas as pd
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
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


if __name__ == "__main__":

    df = load_fraud_data("model_data_v1")
    text_df = separate_text_features_target(df)
    html_features = ["description", "org_desc"]
    text_df = convert_html_to_text(text_df, html_features=html_features)
    print(text_df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        text_df["name"],
        text_df["is_fraud"],
        test_size=0.25,
        random_state=10,
    )

    text_clf_pipeline = Pipeline(
        [
            ("vect", CountVectorizer(ngram_range=(1, 2), stop_words=None)),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier(class_weight="balanced")),
        ]
    )

    text_clf_pipeline.fit(X_train, y_train)

    train_baseline = y_train[y_train == "Not Fraud"].size / y_train.size
    test_baseline = y_test[y_test == "Not Fraud"].size / y_test.size

    train_accuracy = text_clf_pipeline.score(X_train, y_train)
    test_accuracy = text_clf_pipeline.score(X_test, y_test)

    print(f"Train Baseline: {train_baseline:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")

    print(f"Test Baseline: {test_baseline:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
