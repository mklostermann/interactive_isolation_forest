import argparse

import numpy as np
import scipy.stats as st
import os
import re
import helper
import logging


def calc_mean(data, output_file):
    if not data:
        return

    mean = np.mean(data, axis=0)
    # stddev = np.std(data, axis=0)

    cidev = []
    npdata = np.array(data)
    for i in range(0, npdata.shape[1]):  # For every iteration of all runs
        iter_data = npdata[:, i]
        ci = st.norm.interval(confidence=0.95, loc=mean[i], scale=st.sem(iter_data))
        cidev.append(ci[1] - ci[0])
    np.savetxt(output_file, np.array([mean, cidev]), delimiter=',')


def collect_data(data):
    collected_data = []

    for filename in data:
        collected_data.extend(data[filename])

    return collected_data


def main(datasets, algorithms):
    logging.basicConfig(filename="log/calc_metrics.log", filemode='w',
                        format='%(asctime)s [%(threadName)s] %(message)s', level=logging.INFO)
    logging.info("==========")
    logging.info(f"Calculating metrics for data sets {datasets} and algorithms {algorithms}")
    logging.info("==========")

    dataset_infos = helper.get_dataset_info()

    for dataset in datasets:
        logging.info(f"Working on data set {dataset}...")
        labels = {}
        for algorithm in algorithms:
            logging.info(f"Working on algorithm {algorithm} on data set {dataset}")

            results_dir = helper.get_results_dir(dataset, algorithm)
            metrics_dir = helper.get_metrics_dir(dataset, algorithm)

            if not os.path.exists(metrics_dir):
                os.makedirs(metrics_dir)

            if os.path.exists(results_dir):
                anomalies_seen = {}

                for file in os.scandir(results_dir):
                    match = re.search(r"\A(queried_instances)-(\w+)#(\d+)\.csv\Z", file.name)
                    if match is not None:
                        logging.info(f"Calculating metrics for file {file.path}")
                        filename = match.group(2)
                        if filename not in labels:  # Load labels only if there are results for the data set
                            data_file = helper.get_data_file(dataset, filename)
                            logging.info(f"Loading data file {data_file} for {filename}")
                            _, labels[filename] = helper.load_dataset(data_file)

                        if match.group(1) == "queried_instances":
                            # Anomalies seen
                            queried_instances = np.loadtxt(file.path, delimiter=',', dtype=int)
                            value = np.concatenate(([0], np.cumsum(labels[filename][queried_instances])))
                            np.savetxt(os.path.join(metrics_dir, f"anomalies_seen-{filename}#{match.group(3)}.csv"), value, fmt='%d', delimiter=',')

                            if filename not in anomalies_seen:
                                anomalies_seen[filename] = []

                            anomalies_seen[filename].append(value)

                            # TODO: Add metric (or not?)
                            # P@n (n = |O|)
                            # info = next(info for info in dataset_infos if info.filename == filename)
                            # if anomalies_seen.size > info.outlier_count:
                            #     precision_n = anomalies_seen[info.outlier_count] / info.outlier_count
                            #     adjusted_precision_n = (precision_n - (info.outlier_count / info.samples_count))\
                            #                            / (1 - (info.outlier_count / info.samples_count))
                            #     average_precision = sum([(anomalies_seen[i] / i) for i in range(1, info.outlier_count + 1)]) / info.outlier_count
                            #     adjusted_average_precision = (average_precision - (info.outlier_count / info.samples_count))\
                            #                                  / (1 - (info.outlier_count / info.samples_count))
                            #
                            #     np.savetxt(os.path.join(metrics_dir, f"precision-{filename}#{match.group(3)}.csv"),
                            #                [precision_n, adjusted_precision_n, average_precision, adjusted_average_precision], delimiter=',')

                            # R@n (n = |O|)
                    elif file.name.__contains__("omd_summary_feed_"):  # Special handling for OMD
                        omd_anomalies_seen = np.loadtxt(file.path, delimiter=',', dtype=int, skiprows=1)

                        filename = file.name  # TODO: supply dataset file name, e.g. yeast
                        anomalies_seen[filename] = []
                        for i in range(0, omd_anomalies_seen.shape[0]):
                            anomalies_seen[filename].append(omd_anomalies_seen[i][1:])  # Skip first element (iter)
                    else:
                        logging.info(f"File {file.path} is ignored")

                if any(anomalies_seen):
                    logging.info("Calculating mean (...) of seen anomalies")
                    data = collect_data(anomalies_seen)
                    calc_mean(data, os.path.join(metrics_dir, f"anomalies_seen.csv"))

                if not any(anomalies_seen):
                    logging.warning(
                        f"No files to calculate mean of seen anomalies for {dataset}/{algorithm}")
            else:
                logging.warning(f"No result files for {dataset}/{algorithm}")

    logging.info("==========")
    logging.info("Finished")
    logging.info("==========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate metrics from results.")
    parser.add_argument("-a", "--algorithms", type=str, nargs="*",
                        help="algorithms to run the detection on (all if omitted)")
    parser.add_argument("-ds", "--datasets", type=str, nargs="*",
                        help="data sets to run the detection on (all if omitted)")

    args = parser.parse_args()

    algorithms = helper.get_all_algorithms() if args.algorithms is None else args.algorithms
    datasets = helper.get_all_datasets() if args.datasets is None else args.datasets

    try:
        main(datasets, algorithms)
    except:
        logging.exception("Exception in main")
