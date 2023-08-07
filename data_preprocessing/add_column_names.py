import pandas


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
        return None


def add_column_names(dataset):
    try:
        columns = [
            "ID number",
            "Diagnosis",
            "Radius mean",
            "Texture mean",
            "Perimeter mean",
            "Area mean",
            "Smoothness mean",
            "Compactness mean",
            "Concavity mean",
            "Concave points mean",
            "Symmetry mean",
            "Fractal dimension mean",
            "Radius SE",
            "Texture SE",
            "Perimeter SE",
            "Area SE",
            "Smoothness SE",
            "Compactness SE",
            "Concavity SE",
            "Concave points SE",
            "Symmetry SE",
            "Fractal dimension SE",
            "Radius worst",
            "Texture worst",
            "Perimeter worst",
            "Area worst",
            "Smoothness worst",
            "Compactness worst",
            "Concavity worst",
            "Concave points worst",
            "Symmetry worst",
            "Fractal dimension worst"
        ]
        dataset.columns = columns
        print("Column names added.")
        return dataset

    except Exception as e:
        print("Error adding column names: ", e)
        return None


def save_dataset(dataset, new_path):
    try:
        dataset.to_csv(new_path, index=False)
        print(f"Dataset saved as {new_path}")
    except Exception as e:
        print("Error saving the dataset: ", e)
        return None


if __name__ == "__main__":
    """"
    The original dataset has no column names.
    This script adds the column names.
    Found here:
    https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
    """

    # Load the dataset.
    dataset = load_dataset("../datasets/data.csv")
    if dataset is None:
        exit(1)

    # Add the column names.
    dataset = add_column_names(dataset)
    if dataset is None:
        exit(1)

    # Save the dataset.
    save_dataset(dataset, "../datasets/better_data.csv")
