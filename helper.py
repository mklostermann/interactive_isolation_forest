import numpy as np
import pandas as pd
import os


def get_all_datasets():
    return [directory.name for directory in os.scandir(os.path.join(os.getcwd(), "data_sets")) if directory.is_dir()]


def get_data_file(dataset):
    return os.path.abspath(next(file.path for file in os.scandir(f"data_sets/{dataset}") if file.name.endswith(".csv")))


def load_dataset(data_file):
    dataframe = pd.read_csv(data_file)

    data = dataframe.iloc[:, 1:].to_numpy()
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
