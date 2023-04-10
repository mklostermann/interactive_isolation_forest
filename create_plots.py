import argparse

import logging
import matplotlib.pyplot as plt
import pandas as pd

import helper


def plot_auroc(dataset, algorithm_result_file, output_file):
    fig, ax = plt.subplots()

    i = 0
    for (algorithm, result_file) in algorithm_result_file:
        dataframe = pd.read_csv(result_file, header=None)
        data = dataframe.to_numpy(dtype=float)

        ax.errorbar(range(data.shape[1]), data[0], yerr=data[1], label=algorithm, alpha=0.8, elinewidth=0.5, capsize=3,
                    errorevery=[i, len(algorithm_result_file)])
        i = i + 1

    ax.set_xlabel('Iterations')
    ax.set_ylabel('ROC AUC')
    ax.set_title(dataset.upper())
    ax.legend()

    fig.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_anomalies_seen(dataset, algorithm_result_file, output_file):
    fig, ax = plt.subplots()

    i = 0
    for (algorithm, result_file) in algorithm_result_file:
        dataframe = pd.read_csv(result_file, header=None)
        data = dataframe.to_numpy(dtype=float)
        ax.errorbar(range(data.shape[1]), data[0], yerr=data[1], label=algorithm, alpha=0.8, elinewidth=0.5, capsize=3,
                    errorevery=[i, len(algorithm_result_file)])
        i = i + 1

    ax.set_xlabel('Iterations')
    ax.set_ylabel('# Anomalies')
    ax.set_title(dataset.upper())
    ax.legend()

    fig.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(fig)


def main(datasets, algorithms):
    logging.basicConfig(filename="log/create_plots.log", filemode='w',
                        format='%(asctime)s [%(threadName)s] %(message)s', level=logging.INFO)
    logging.info("==========")
    logging.info(f"Creating plots for data sets {datasets} and algorithms {algorithms}")
    logging.info("==========")

    # TODO: Remove stuff to handle variants (configurable)
    variants = ["all", "le05", "le10", "le20", "orig", "02", "05", "10", "20"]

    for dataset in datasets:
        logging.info(f"Creating plots for {dataset}")

        name = "auroc"
        algorithm_result_file = helper.get_metrics_files_for_algorithms(dataset, algorithms, f"{name}.csv")
        if any(algorithm_result_file):
            plot_auroc(dataset, algorithm_result_file, helper.get_plot_file(dataset, name))

        name = "anomalies_seen"
        algorithm_result_file = helper.get_metrics_files_for_algorithms(dataset, algorithms, f"{name}.csv")
        if any(algorithm_result_file):
            plot_anomalies_seen(dataset, algorithm_result_file, helper.get_plot_file(dataset, name))

    logging.info("==========")
    logging.info("Finished")
    logging.info("==========")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs anomaly detection.")
    parser.add_argument("-a", "--algorithms", type=str, nargs="*",
                        help="algorithms to run the detection on (all if omitted)")
    parser.add_argument("-ds", "--datasets", type=str, nargs="*",
                        help="data sets to run the detection on (all if omitted)")

    args = parser.parse_args()

    if args.algorithms is None:
        args.algorithms = helper.get_all_algorithms()

    if args.datasets is None:
        args.datasets = helper.get_all_datasets()

    main(args.datasets, args.algorithms)
