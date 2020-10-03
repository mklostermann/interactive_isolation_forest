import numpy as np
import pandas as pd
import scipy.io
import os
from scipy.io import arff


def get_all_datasets():
    return [directory.name for directory in os.scandir(os.path.join(os.getcwd(), "data_sets"))
            if directory.is_dir() and not directory.name.startswith("__")]


def get_data_file(dataset):
    return get_all_data_files(dataset, ".csv")[0]  # TODO: Implement something like get_next_data_file


def get_all_data_files(dataset, extension):
    return [os.path.abspath(file.path) for file in os.scandir(f"data_sets/{dataset}") if file.name.endswith(extension)]


def load_dataset(data_file):
    dataframe = pd.read_csv(data_file)

    data = dataframe.iloc[:, 1:].to_numpy(dtype=float)
    labels = np.array([1 if dataframe.iloc[i, 0] == "anomaly" else 0 for i in range(dataframe.shape[0])], dtype=int)

    return data, labels


def get_all_algorithms():
    return [directory.name for directory in os.scandir(os.path.join(os.getcwd(), "algorithms"))
            if directory.is_dir() and not directory.name.startswith("__")]


def get_results_dir(dataset, algorithm=None):
    return os.path.abspath(os.path.join(os.getcwd(), f"results/{dataset}/{'' if algorithm is None else algorithm}"))


def get_plot_file(dataset, plotname):
    return os.path.abspath(os.path.join(os.getcwd(), f"plots/{dataset}-{plotname}.pdf"))


def get_result_files_for_algorithms(dataset, algorithms, name):
    results_dir = get_results_dir(dataset)

    files = [(algorithm, os.path.join(results_dir, algorithm, name)) for algorithm in algorithms]
    return list(filter(lambda x: os.path.exists(x[1]), files))


# TODO: Remove dead code
def convert_mat_to_csv(dataset):
    orig = scipy.io.loadmat(f"data_sets/{dataset}/{dataset}.mat")
    data = orig['X']
    labels = ["anomaly" if label == 1 else "nominal" for label in orig['y']]

    full_data = {"labels": labels}
    for i in range(data.shape[1]):
        full_data[f"x{i}"] = data[:, i]
    dataframe = pd.DataFrame(full_data)

    dataframe.to_csv(f"data_sets/{dataset}/{dataset}.csv", index=False)


def convert_all_arff_to_csv():
    for dataset in get_all_datasets():
        for file in get_all_data_files(dataset, ".arff"):
            print(f"Converting data file {file}...")

            # Load and convert data
            dataframe = pd.DataFrame(arff.loadarff(file)[0])
            data = {}
            i = 0
            for column in dataframe.columns:
                if column.lower() == "id":
                    continue
                if column.lower() == "outlier":
                    labels = ["anomaly" if outlier == b"yes" else "nominal" for outlier in dataframe[column].values]
                    continue

                if column != "id" and column != "outlier":
                    data[f"{column}"] = dataframe[column].values
                    i = i + 1

            # Merge data and store as CSV
            full_data = {"labels": labels}
            for attribute in data:
                full_data[attribute] = data[attribute]

            csv_file = file.replace(".arff", ".csv")
            pd.DataFrame(full_data).to_csv(csv_file, index=False)

            print(f"Succeeded, result was stored at {csv_file}")
