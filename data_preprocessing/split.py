import pandas
from describe import describe


def load_dataset(path):
    """
    Load the dataset.
    """
    try:
        dataset = pandas.read_csv(path)
        print("Dataset loaded.")
        return dataset
    except Exception:
        print("Error: Dataset not found.")
        exit()


def add_column_names(dataset, columns):
    try:
        dataset.columns = columns
        print("Column names added.")
        return dataset

    except Exception as e:
        print("Error adding column names: ", e)
        exit()


def split_dataset(dataset, train_percentage):
    try:
        train = dataset.sample(frac=train_percentage, random_state=0)
        validation = dataset.drop(train.index)
        print("Dataset splitted.")
        print(f"Train set: {len(train)} samples.")
        print(f"Validation set: {len(validation)} samples.")
        return train, validation
    except Exception as e:
        print("Error splitting the dataset: ", e)
        exit()


def save_dataset(dataset, new_path):
    try:
        dataset.to_csv(new_path, index=False)
        print(f"Dataset saved as {new_path}")
    except Exception as e:
        print("Error saving the dataset: ", e)
        exit()


if __name__ == "__main__":
    """"
    The original dataset has no column names.

    The labels was found here:
    https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

    Add the column names and
    split the dataset into a train and a validation set.
    """

    features = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave points",
        "Symmetry",
        "Fractal dimension"
    ]

    # The 10 first features are the mean of the cells.
    # The 10 next features are the standard error of the cells.
    # The 10 last features are the worst (mean of the three largest values)
    # of the cells.
    mean = [f"{feature} mean" for feature in features]
    se = [f"{feature} SE" for feature in features]
    worst = [f"{feature} worst" for feature in features]

    # The first two columns are the ID number and the diagnosis.
    columns = ["ID number", "Diagnosis"] + mean + se + worst

    # Load the original dataset.
    dataset = load_dataset("../datasets/data.csv")

    # Add the column names.
    dataset = add_column_names(dataset, columns)

    # Drop the 'ID number' column.
    dataset = dataset.drop("ID number", axis=1)

    describe(dataset)

    # Split the dataset.
    train, validation = split_dataset(dataset, 0.8)

    # Save the train and the validation datasets.
    save_dataset(train, "../datasets/train.csv")
    save_dataset(validation, "../datasets/validation.csv")
