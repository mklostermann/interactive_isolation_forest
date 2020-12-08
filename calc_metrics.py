import numpy as np
import os
import re
import helper
import logging

from sklearn import metrics


def calc_auroc(labels, input_file, output_file):
    all_scores = np.loadtxt(input_file, delimiter=',')
    auroc = [metrics.roc_auc_score(labels, all_scores[i, :]) for i in range(all_scores.shape[0])]
    np.savetxt(output_file, auroc, delimiter=',')


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
                            calc_auroc(labels[filename], file.path,
                                       os.path.join(metrics_dir,
                                                    f"auroc-{filename}#{match.group(3)}.csv"))

                        if match.group(1) == "queried_instances":
                            # Anomalies seen
                            queried_instances = np.loadtxt(file.path, delimiter=',', dtype=int)
                            anomalies_seen = np.concatenate(([0], np.cumsum(labels[filename][queried_instances])))
                            np.savetxt(os.path.join(metrics_dir, f"anomalies_seen-{filename}#{match.group(3)}.csv"), anomalies_seen, fmt='%d', delimiter=',')

                            # P@n
                            info = next(info for info in dataset_infos if info.filename == filename)
                            if anomalies_seen.size > info.outlier_count:
                                precision_n = anomalies_seen[info.outlier_count] / info.outlier_count
                                adjusted_precision_n = (precision_n - (info.outlier_count / info.samples_count))\
                                                       / (1 - (info.outlier_count / info.samples_count))
                                average_precision = sum([(anomalies_seen[i] / i) for i in range(1, info.outlier_count + 1)]) / info.outlier_count
                                adjusted_average_precision = (average_precision - (info.outlier_count / info.samples_count))\
                                                             / (1 - (info.outlier_count / info.samples_count))

                                np.savetxt(os.path.join(metrics_dir, f"precision-{filename}#{match.group(3)}.csv"),
                                           [precision_n, adjusted_precision_n, average_precision, adjusted_average_precision], delimiter=',')

                    else:
                        logging.info(f"File {file.path} is ignored")
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
