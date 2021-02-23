import pandas as pd
from sklearn.model_selection import train_test_split


def working_validation_split(
    full_data_filepath, working_data_filepath, validation_data_filepath
):
    """
    Splits JSON file into working and validation sets. 80/20 split.

    Args:
        full_data_filepath (string)
        working_data_filepath (string)
        validation_data_filepath (string)
    """
    full_data = pd.read_json(full_data_filepath, orient="records")
    working_data, validation_data = train_test_split(full_data, test_size=0.20)
    working_data.to_json(working_data_filepath, orient="records", lines=True)
    validation_data.to_json(
        validation_data_filepath, orient="records", lines=True
    )


if __name__ == "__main__":
    full_data_path = "../data/data.json"
    working_data_path = "../data/working_data.json"
    validation_data_path = "../data/validation_data.json"
    working_validation_split(
        full_data_path, working_data_path, validation_data_path
    )
