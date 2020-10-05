import numpy as np
import os
import re
import helper

from sklearn import metrics

def calc_auroc(labels, input_file, output_file):
    all_scores = np.loadtxt(input_file, delimiter=',')
    auroc = [metrics.roc_auc_score(labels, all_scores[i, :]) for i in range(all_scores.shape[0])]
    np.savetxt(output_file, auroc, fmt='%f', delimiter=',')


def calc_anomalies_seen(labels, input_file, output_file):
    queried_instances = np.loadtxt(input_file, delimiter=',', dtype=int)
    anomalies_seen = np.concatenate(([0], np.cumsum(labels[queried_instances])))
    np.savetxt(output_file, anomalies_seen, fmt='%d', delimiter=',')


# TODO: Create averaged scores
def main(datasets, algorithms):
    for dataset in datasets:
        labels = None

        for algorithm in algorithms:
            results_dir = helper.get_results_dir(dataset, algorithm)

            if os.path.exists(results_dir):
                if labels is None:  # Load labels only if there are results for the data set
                    _, labels = helper.load_dataset(helper.get_data_file(dataset))

                for file in os.scandir(results_dir):
                    match = re.search(r"\Aall_scores-(\w+#\d+)\.csv\Z", file.name)
                    if match is not None:
                        calc_auroc(labels, file.path, os.path.join(results_dir, f"auroc-{match.group(1)}.csv"))
                    match = re.search(r"\Aqueried_instances-(\w#\d+)\.csv\Z", file.name)
                    if match is not None:
                        calc_anomalies_seen(labels, file.path, os.path.join(results_dir, f"anomalies_seen-{match.group(1)}.csv"))


if __name__ == '__main__':
    main(helper.get_all_datasets(), helper.get_all_algorithms())
