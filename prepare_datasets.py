"""
Prepares all datasets in the './datasets' folder:
 - Converts *.arff to *.csv
 - Collects information (N, outliers, outlier rate...) about each *.csv file
 - Collected information is stored in './datasets/info.json'
"""

import pandas as pd
import numpy as np
import helper
import json
import os
import re
from scipy.io import arff


def convert_all_arff_to_csv():
    for dataset in helper.get_all_datasets():
        for file in helper.get_all_data_files(dataset, ".arff"):
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

    @classmethod
    def fromfile(cls, dataset, filepath):
        data, labels = helper.load_dataset(filepath)

        filepath = os.path.relpath(filepath)
        filename = helper.get_filename(filepath)
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
    result = []

    for dataset in helper.get_all_datasets():
        for file in helper.get_all_data_files(dataset):
            result.append(DatasetInfo.fromfile(dataset, file))

    return result


def main():
    print("==== Converting all datasets to *.csv ====")
    convert_all_arff_to_csv()

    print("==== Collecting dataset information ====")
    info = get_dataset_info()

    print("==== Storing info.json ====")
    with open(helper.get_dataset_info_file(), 'w') as file:
        json.dump(info, file, indent=4, default=lambda obj: obj.__dict__)


if __name__ == '__main__':
    main()
