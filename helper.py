import numpy as np
import pandas as pd
import json
import os
import re


class DatasetInfo():
    def __init__(self, dataset, filename, filepath, samples_count, outlier_count, outlier_rate, filename_outlier_rate, attribute_count,
               without_duplicates, normalized, downsampled_with_variations):
        self.dataset = dataset
        self.filename = filename
        self.filepath = filepath
        self.samples_count = samples_count
        self.outlier_count = outlier_count
        self.filename_outlier_rate = filename_outlier_rate # Outlier rate according to dataset file name (rounded; calculated without duplicates)
        self.outlier_rate = outlier_rate
        self.attribute_count = attribute_count
        self.without_duplicates = without_duplicates
        self.normalized = normalized
        self.downsampled_with_variations = downsampled_with_variations

    def get_data_file(self):
        return os.path.abspath(self.filepath)

    @classmethod
    def fromfile(cls, dataset, filepath):
        data, labels = load_dataset(filepath)

        filepath = os.path.relpath(filepath)
        filename = get_filename(filepath)
        samples_count = labels.shape[0]
        outlier_count = int(np.sum(labels))
        outlier_rate = outlier_count / samples_count * 100
        attribute_count = data.shape[1]
        without_duplicates = "_withoutdupl" in filename
        normalized = "_norm" in filename

        match = re.search(r"_(\d\d)", filename)
        filename_outlier_rate = int(match.group(1)) if match is not None else None

        match = re.search(r"(_v\d\d)", filename)
        downsampled_with_variations = match is not None

        assert "-" not in filename, \
            "Dataset file name must not contain '-' (required to parse result/metric files)"

        return cls(dataset, filename, filepath, samples_count, outlier_count, outlier_rate, filename_outlier_rate,
                   attribute_count, without_duplicates, normalized, downsampled_with_variations)

    @classmethod
    def fromdict(cls, dict):
        return cls(dict["dataset"], dict["filename"], dict["filepath"], dict["samples_count"], dict["outlier_count"],
                   dict["outlier_rate"], dict["filename_outlier_rate"], dict["attribute_count"], dict["without_duplicates"],
                   dict["normalized"], dict["downsampled_with_variations"])


def get_dataset_info():
    with open(get_dataset_info_file(), 'r') as file:
        return json.load(file, object_hook=lambda dct: DatasetInfo.fromdict(dct))


def get_all_datasets():
    return [directory.name for directory in os.scandir(os.path.join(os.getcwd(), "datasets"))
            if directory.is_dir() and not directory.name.startswith("__")]


def get_data_file(dataset, filename):
    return os.path.abspath(f"datasets/{dataset}/{filename}.csv")


def get_dataset_info_file():
    return os.path.abspath("datasets/info.json")


def get_outlier_rate(data_file_name): # TODO: remove (was moved to prepare_datasets, use DatasetInfo instead)
    outlier_rate = None
    downsampled = False

    match = re.search(r"_(\d\d)_(v?)", data_file_name)
    if match is not None:
        outlier_rate = int(match.group(1))
        downsampled = match.group(2) is not None

    return outlier_rate, downsampled


def get_all_data_files(dataset, extension=".csv"):
    return [os.path.abspath(file.path) for file in os.scandir(f"datasets/{dataset}") if file.name.endswith(extension)]


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


def get_metrics_dir(dataset, algorithm=None):
    return os.path.abspath(os.path.join(os.getcwd(), f"metrics/{dataset}/{'' if algorithm is None else algorithm}"))


def get_plot_file(dataset, algorithms, plotname, extension=".pdf"):
    return os.path.abspath(os.path.join(os.getcwd(), f"plots/{dataset}-{'_'.join(sorted(algorithms))}-{plotname}{extension}"))


def get_metrics_files_for_algorithms(dataset, algorithms, name):
    metrics_dir = get_metrics_dir(dataset)

    files = [(algorithm, os.path.join(metrics_dir, algorithm, name)) for algorithm in algorithms]
    return list(filter(lambda x: os.path.exists(x[1]), files))


def get_filename(path):
    return os.path.basename(path).rsplit(".", 1)[0]

def save_queried_instances(queried_instances, results_dir, data_file, run):
    queried_file = os.path.join(results_dir, f"queried_instances-{get_filename(data_file)}#{run}.csv")
    np.savetxt(queried_file, queried_instances, fmt="%d", delimiter=",")

def save_active_trees(active_trees, results_dir, data_file, run):
    file = os.path.join(results_dir, f"active_trees-{get_filename(data_file)}#{run}.csv")
    np.savetxt(file, active_trees, fmt="%d", delimiter=",")

def save_trained_trees(trained_trees, results_dir, data_file, run):
    file = os.path.join(results_dir, f"trained_trees-{get_filename(data_file)}#{run}.csv")
    np.savetxt(file, trained_trees, fmt="%d", delimiter=",")
