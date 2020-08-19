import numpy as np
import os
import re
import helper

from sklearn import metrics
import matplotlib.pyplot as plt
import itertools


def plot_auroc(dataset, algorithm_result_file, output_file):
    fig, ax = plt.subplots()

    for (algorithm, result_file) in algorithm_result_file:
        data = np.loadtxt(result_file)
        ax.plot(range(data.shape[0]), data, label=algorithm, alpha=0.8)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('ROC AUC')
    ax.set_title(dataset.upper())
    ax.legend()

    fig.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_anomalies_seen(dataset, algorithm_result_file, output_file):
    fig, ax = plt.subplots()

    for (algorithm, result_file) in algorithm_result_file:
        data = np.loadtxt(result_file, dtype=int)
        ax.plot(range(data.shape[0]), data, label=algorithm, alpha=0.8)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('# Anomalies')
    ax.set_title(dataset.upper())
    ax.legend()

    fig.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(fig)


def main(datasets, algorithms):
    for dataset in datasets:

        for i in range(1, 100): # TODO
            algorithm_result_file = helper.get_result_files_for_algorithms(dataset, algorithms, f"auroc-{i}.csv")
            if any(algorithm_result_file):
                plot_auroc(dataset, algorithm_result_file, helper.get_plot_file(dataset, f"auroc-{i}"))
            else:
                break

        for i in range(1, 100):  # TODO
            algorithm_result_file = helper.get_result_files_for_algorithms(dataset, algorithms, f"anomalies_seen-{i}.csv")
            if any(algorithm_result_file):
                plot_anomalies_seen(dataset, algorithm_result_file, helper.get_plot_file(dataset, f"anomalies_seen-{i}"))
            else:
                break


if __name__ == '__main__':
    main(helper.get_all_datasets(), helper.get_all_algorithms())
