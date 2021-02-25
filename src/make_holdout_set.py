import pandas as pd
from sklearn.model_selection import train_test_split


def working_holdout_split(
    full_data_filepath, working_data_filepath, holdout_data_filepath
):
    """
    Splits JSON file into working and validation sets. 80/20 split.

    Args:
        full_data_filepath (string)
        working_data_filepath (string)
        validation_data_filepath (string)
    """
    full_data = pd.read_json(full_data_filepath, orient="records")
    working_data, holdout_data = train_test_split(full_data, test_size=0.20)
    working_data.to_json(working_data_filepath, orient="records", lines=True)
    holdout_data.to_json(holdout_data_filepath, orient="records", lines=True)


if __name__ == "__main__":
    full_data_path = "../data/data.json"
    working_data_path = "../data/working_data.json"
    holdout_data_path = "../data/holdout_data.json"
    working_holdout_split(full_data_path, working_data_path, holdout_data_path)
