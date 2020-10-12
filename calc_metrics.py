import numpy as np
import os
import re
import helper

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
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    np.savetxt(output_file, mean, fmt='%f', delimiter=',')


# TODO: Create averaged scores
def main(datasets, algorithms):
    for dataset in datasets:
        labels = {}
        for algorithm in algorithms:
            results_dir = helper.get_results_dir(dataset, algorithm)

            if os.path.exists(results_dir):
                auroc = {}
                anomalies_seen = {}

                for file in os.scandir(results_dir):
                    match = re.search(r"\A(all_scores|queried_instances)-(\w+)#(\d+)\.csv\Z", file.name)
                    if match is not None:
                        filename = match.group(2)
                        if filename not in labels:  # Load labels only if there are results for the data set
                            _, labels[filename] = helper.load_dataset(helper.get_data_file(dataset, filename))

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

                all_data = []
                for filename in auroc:
                    calc_mean(auroc[filename],
                              os.path.join(results_dir, f"auroc-{filename}.csv"))
                    all_data.extend(auroc[filename])

                calc_mean(all_data, os.path.join(results_dir, "auroc.csv"))


                all_data = []
                for filename in anomalies_seen:
                    calc_mean(anomalies_seen[filename],
                              os.path.join(results_dir, f"anomalies_seen-{filename}.csv"))
                    all_data.extend(anomalies_seen[filename])

                calc_mean(all_data, os.path.join(results_dir, "anomalies_seen.csv"))


if __name__ == '__main__':
    main(helper.get_all_datasets(), helper.get_all_algorithms())
