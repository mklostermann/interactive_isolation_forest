import numpy as np
import os
import re
import helper
import logging

from sklearn import metrics


def calc_auroc(labels, input_file, output_file):
    all_scores = np.loadtxt(input_file, delimiter=',')
    auroc = [metrics.roc_auc_score(labels, all_scores[i, :]) for i in range(all_scores.shape[0])]
    np.savetxt(output_file, auroc, fmt='%f', delimiter=',')
    return auroc


def calc_anomalies_seen(labels, input_file, output_file):
    queried_instances = np.loadtxt(input_file, delimiter=',', dtype=int)
    anomalies_seen = np.concatenate(([0], np.cumsum(labels[queried_instances])))
    np.savetxt(output_file, anomalies_seen, fmt='%d', delimiter=',')
    return anomalies_seen


def calc_mean(data, output_file):
    if not data:
        return

    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    np.savetxt(output_file, np.array([mean, stddev]), fmt='%f', delimiter=',')


def collect_data(data):
    collected_data = {"all": [], "le05": [], "le10": [], "le20": [], "orig": [], "02": [], "05": [], "10": [], "20": []}

    for filename in data:
        collected_data["all"].extend(data[filename])

        outlier_rate, downsampled = helper.get_outlier_rate(filename)
        if not downsampled:
            collected_data["orig"].extend(data[filename])
        if outlier_rate is not None:
            if outlier_rate == 2:
                collected_data["02"].extend(data[filename])
            if outlier_rate == 5:
                collected_data["05"].extend(data[filename])
            if outlier_rate == 10:
                collected_data["10"].extend(data[filename])
            if outlier_rate == 20:
                collected_data["20"].extend(data[filename])
            if outlier_rate <= 5:
                collected_data["le05"].extend(data[filename])
            if outlier_rate <= 10:
                collected_data["le10"].extend(data[filename])
            if outlier_rate <= 20:
                collected_data["le20"].extend(data[filename])

    return collected_data


# TODO: Create averaged scores
def main(datasets, algorithms):
    logging.basicConfig(filename="log/calc_metrics.log", filemode='w',
                        format='%(asctime)s [%(threadName)s] %(message)s', level=logging.INFO)
    logging.info("==========")
    logging.info(f"Calculating metrics for data sets {datasets} and algorithms {algorithms}")
    logging.info("==========")

    for dataset in datasets:
        logging.info(f"Working on data set {dataset}...")
        labels = {}
        for algorithm in algorithms:
            logging.info(f"Working on algorithm {algorithm} on data set {dataset}")

            results_dir = helper.get_results_dir(dataset, algorithm)

            if os.path.exists(results_dir):
                auroc = {}
                anomalies_seen = {}

                for file in os.scandir(results_dir):
                    match = re.search(r"\A(all_scores|queried_instances)-(\w+)#(\d+)\.csv\Z", file.name)
                    if match is not None:
                        logging.info(f"Calculating metrics for file {file.path}")
                        filename = match.group(2)
                        if filename not in labels:  # Load labels only if there are results for the data set
                            data_file = helper.get_data_file(dataset, filename)
                            logging.info(f"Loading data file {data_file} for {filename}")
                            _, labels[filename] = helper.load_dataset(data_file)

                        if match.group(1) == "all_scores":
                            value = calc_auroc(labels[filename], file.path,
                                               os.path.join(results_dir,
                                                            f"auroc-{match.group(2)}#{match.group(3)}.csv"))
                            if filename not in auroc:
                                auroc[filename] = []

                            auroc[filename].append(value)

                        if match.group(1) == "queried_instances":
                            value = calc_anomalies_seen(labels[filename], file.path,
                                                        os.path.join(results_dir,
                                                                     f"anomalies_seen-{match.group(2)}#{match.group(3)}.csv"))

                            if filename not in anomalies_seen:
                                anomalies_seen[filename] = []

                            anomalies_seen[filename].append(value)
                    else:
                        logging.info(f"File {file.path} is ignored")

                if any(auroc) and any(anomalies_seen):
                    logging.info("Calculating mean (...) of AUROC")
                    data = collect_data(auroc)

                    for key in data:
                        calc_mean(data[key], os.path.join(results_dir, f"auroc-{key}.csv"))

                    logging.info("Calculating mean (...) of seen anomalies")
                    data = collect_data(anomalies_seen)

                    for key in data:
                        calc_mean(data[key], os.path.join(results_dir, f"anomalies_seen-{key}.csv"))
                else:
                    logging.warning(f"No files to calculate mean of AUROC / seen anomalies for {dataset}/{algorithm}")
            else:
                logging.warning(f"No result files for {dataset}/{algorithm}")

    logging.info("==========")
    logging.info("Finished")
    logging.info("==========")


if __name__ == '__main__':
    try:
        main(helper.get_all_datasets(), helper.get_all_algorithms())
    except:
        logging.exception("Exception in main")
