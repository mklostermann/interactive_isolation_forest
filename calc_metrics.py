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


def calc_anomalies_seen(labels, input_file, output_file):
    queried_instances = np.loadtxt(input_file, delimiter=',', dtype=int)
    anomalies_seen = np.concatenate(([0], np.cumsum(labels[queried_instances])))
    np.savetxt(output_file, anomalies_seen, fmt='%d', delimiter=',')


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
            metrics_dir = helper.get_metrics_dir(dataset, algorithm)

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
                                                    f"auroc-{match.group(2)}#{match.group(3)}.csv"))

                        if match.group(1) == "queried_instances":
                            calc_anomalies_seen(labels[filename], file.path,
                                                os.path.join(metrics_dir,
                                                             f"anomalies_seen-{match.group(2)}#{match.group(3)}.csv"))
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
