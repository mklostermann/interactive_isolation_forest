import numpy as np
import pandas as pd
import scipy.io
import os
import re


def get_all_datasets():
    return [directory.name for directory in os.scandir(os.path.join(os.getcwd(), "datasets"))
            if directory.is_dir() and not directory.name.startswith("__")]


def get_data_file(dataset, filename):
    return os.path.abspath(f"datasets/{dataset}/{filename}.csv")


def get_dataset_info_file():
    return os.os.path.abspath("datasets/info.json")


def get_outlier_rate(data_file_name):
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


def get_plot_file(dataset, plotname):
    return os.path.abspath(os.path.join(os.getcwd(), f"plots/{dataset}-{plotname}.pdf"))


def get_result_files_for_algorithms(dataset, algorithms, name):
    results_dir = get_results_dir(dataset)

    files = [(algorithm, os.path.join(results_dir, algorithm, name)) for algorithm in algorithms]
    return list(filter(lambda x: os.path.exists(x[1]), files))


# TODO: Remove dead code
def convert_mat_to_csv(dataset):
    orig = scipy.io.loadmat(f"datasets/{dataset}/{dataset}.mat")
    data = orig['X']
    labels = ["anomaly" if label == 1 else "nominal" for label in orig['y']]

    full_data = {"labels": labels}
    for i in range(data.shape[1]):
        full_data[f"x{i}"] = data[:, i]
    dataframe = pd.DataFrame(full_data)

    dataframe.to_csv(f"datasets/{dataset}/{dataset}.csv", index=False)





def get_filename(path):
    return os.path.basename(path).rsplit(".", 1)[0]


def save_queried_instances(queried_instances, results_dir, data_file, run):
    queried_file = os.path.join(results_dir, f"queried_instances-{get_filename(data_file)}#{run}.csv")
    np.savetxt(queried_file, queried_instances, fmt="%d", delimiter=",")


def save_all_scores(all_scores, results_dir, data_file, run):
    all_scores_file = os.path.join(results_dir, f"all_scores-{get_filename(data_file)}#{run}.csv")
    np.savetxt(all_scores_file, all_scores, fmt='%f', delimiter=',')