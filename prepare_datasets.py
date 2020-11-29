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


def get_dataset_info():
    result = []

    for dataset in helper.get_all_datasets():
        for file in helper.get_all_data_files(dataset):
            result.append(helper.DatasetInfo.fromfile(dataset, file))

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
