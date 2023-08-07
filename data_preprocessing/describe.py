from argparse import ArgumentParser
import pandas
from metrics import TinyStatistician as Metrics
from os import get_terminal_size


def parse_arguments() -> tuple:
    """
    Parse the command line argument.
    Positional argument:
    - The program takes one positional argument, the path of the dataset.
    Optional arguments:
    - Bonus : [--bonus | -b] display more metrics.
    - Compare : [--compare | -c] compare with real describe().
    - Help : [--help | -h] display an help message.
    Usage:
      python describe.py [-b | --bonus] [-c | --compare] [-h | --help] data.csv
    """
    try:
        parser = ArgumentParser(
            prog="describe",
            description="This program takes a dataset path as argument. " +
            "It displays informations for all numerical features."
        )
        parser.add_argument(
            dest="dataset_path",
            type=str,
            help="Path to the dataset."
        )
        args = parser.parse_args()
        return args.dataset_path
    except Exception as e:
        print("Error parsing arguments: ", e)
        exit()


def read_dataset(dataset_path: str) -> pandas.DataFrame:
    """
    Read the dataset from the given path,
    returned as a pandas DataFrame.
    """
    try:
        dataset = pandas.read_csv(dataset_path)
        if dataset.empty:
            print("The dataset is empty.")
            return None
        return dataset
    except FileNotFoundError:
        print("Error: dataset not found.")
        exit()
    except Exception as e:
        print("Error reading dataset: ", e)
        exit()


def select_columns(dataset: pandas.DataFrame) -> pandas.DataFrame:
    """
    Describe display numerical features metrics.
    Select only the numerical columns, drop the Index.
    """
    try:
        numerical_dataset = dataset.select_dtypes(include="number")
        if numerical_dataset.empty:
            print("The dataset does not contain numerical features.")
            return None, None
        numerical_dataset = numerical_dataset.drop("ID number", axis=1)
        columns = numerical_dataset.columns
        return (
            numerical_dataset,
            columns
        )

    except Exception as e:
        print("Error selecting columns: ", e)
        exit()


def describe(dataset_path: str):
    """
    Describe display numerical features metrics.
    Arguments:
    - dataset_path: path to the dataset.
    - full_metrics: display more metrics.
    """
    try:

        # Read the dataset.
        entire_dataset = read_dataset(dataset_path)
        if entire_dataset is None:
            return None

        # Select only the numerical columns, drop the Index.
        dataset, feature_names = select_columns(entire_dataset)
        if dataset is None or feature_names is None:
            return None

        metrics = {
            "count": Metrics.count,
            "mean": Metrics.mean,
            "mode": Metrics.mode,
            "var": Metrics.var,
            "std": Metrics.std,
            "min": Metrics.min,
            "25%": Metrics.perc25,
            "50%": Metrics.perc50,
            "75%": Metrics.perc75,
            "max": Metrics.max,
            "range": Metrics.range,
            "iqr": Metrics.iqr,
            "aad": Metrics.aad,
            "cv": Metrics.cv,
        }

        # Create a DataFrame to store the metrics.
        description = pandas.DataFrame(
            index=metrics.keys(),
            columns=feature_names,
            dtype=float,
        )

        # Compute the metrics for each feature.
        for feature in feature_names:
            np_feature = dataset[feature].dropna().to_numpy()
            for metric, function in metrics.items():
                description.loc[metric, feature] = function(np_feature)

        # Display the DataFrame.
        with pandas.option_context(
            'display.max_columns', None,
            'display.width', get_terminal_size().columns
        ):
            print(description)

        return description

    except Exception as error:
        print("Error: ", error)
        return None


if __name__ == "__main__":
    dataset_path = parse_arguments()
    describe(dataset_path)
