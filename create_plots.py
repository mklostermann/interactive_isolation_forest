import argparse

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helper

def get_label(algorithm):
    if algorithm == "iforest":
        return "IF-OT"
    elif algorithm == "iforest_rep":
            return "IF"
    elif algorithm == "iforest_rep_dp":
        return "IF-DP"
    elif algorithm == "tiws_if":
        return "TIWS-OT"
    elif algorithm == "tiws_if_rep":
        return "TIWS"
    elif algorithm == "tiws_if_replace2_fp":
        return "TIWS-OTR-FP"
    elif algorithm == "omd":
        return "OMD"
    elif algorithm == "iforest_aad":
        return "IF-AAD"
    elif algorithm == "tiws_if_replace2":
        return "TIWS-OTR"
    elif algorithm == "tiws_if_replace2_dp":
        return "TIWS-OTR-DP (IIF)"
    else:
        return algorithm

def get_color(algorithm):
    # Default cycle: tab:blue-, tab:orange-, tab:green-, tab:red-, tab:purple-, tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan
    if algorithm == "iforest":
        return "tab:brown"
    elif algorithm == "iforest_rep":
        return "tab:blue"
    elif algorithm == "iforest_rep_dp":
        return "tab:gray"
    elif algorithm == "tiws_if":
        return "tab:brown"
    elif algorithm == "tiws_if_rep":
        return "tab:orange"
    elif algorithm == "tiws_if_replace2_fp":
        return "tab:gray"
    elif algorithm == "omd":
        return "tab:green"
    elif algorithm == "iforest_aad":
        return "tab:purple"
    elif algorithm == "tiws_if_replace2":
        return "tab:brown"
    elif algorithm == "tiws_if_replace2_dp":
        return "tab:red"
    else:
        return "magenta"


def plot_anomalies_seen(dataset, algorithms, algorithm_result_file, output_file):
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", map(get_color, algorithms))

    i = 0
    for (algorithm, result_file) in algorithm_result_file:
        dataframe = pd.read_csv(result_file, header=None)
        data = dataframe.to_numpy(dtype=float)
        ax.errorbar(range(data.shape[1]), data[0], yerr=data[1], label=get_label(algorithm), alpha=0.8, elinewidth=0.5, capsize=3,
                    errorevery=(i, len(algorithm_result_file)))
        i = i + 1

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Number of True Anomalies')
    # TODO: Set scale? ax.set_ylim([0, data.shape[1]])
    # No title as I am using captions: ax.set_title(dataset.upper())
    ax.legend()

    fig.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_active_trees(dataset, algorithms, algorithm_result_file, output_file):
    fig, ax = plt.subplots()
    ax.set_prop_cycle("color", map(get_color, algorithms))

    i = 0
    for (algorithm, result_file) in algorithm_result_file:
        dataframe = pd.read_csv(result_file, header=None)
        data = dataframe.to_numpy(dtype=float)
        data[:, 0] = np.nan # Skip first value, there is no tree at iteration 0
        ax.errorbar(range(data.shape[1]), data[0], yerr=data[1], label=get_label(algorithm), alpha=0.8, elinewidth=0.5, capsize=3,
                    errorevery=(i, len(algorithm_result_file)))
        i = i + 1

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Forest Size')
    # TODO: Set scale? ax.set_ylim([0, data.shape[1]])
    # No title as I am using captions: ax.set_title(dataset.upper())
    ax.legend()

    fig.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(fig)


def main(datasets, algorithms):
    logging.basicConfig(filename="log/create_plots.log", filemode='w',
                        format='%(asctime)s [%(threadName)s] %(message)s', level=logging.INFO)
    logging.info("==========")
    logging.info(f"Creating plots for data sets {datasets} and algorithms {algorithms}")
    logging.info("==========")

    for dataset in datasets:
        logging.info(f"Creating plots for {dataset}")

        name = "anomalies_seen"
        algorithm_result_file = helper.get_metrics_files_for_algorithms(dataset, algorithms, f"{name}.csv")
        present_algorithms = [i[0] for i in algorithm_result_file]
        if any(algorithm_result_file):
            plot_anomalies_seen(dataset, present_algorithms, algorithm_result_file,
                                helper.get_plot_file(dataset, present_algorithms, name))

        name = "active_trees"
        algorithm_result_file = helper.get_metrics_files_for_algorithms(dataset, algorithms, f"{name}.csv")
        present_algorithms = [i[0] for i in algorithm_result_file]
        if any(algorithm_result_file):
            plot_active_trees(dataset, present_algorithms, algorithm_result_file,
                              helper.get_plot_file(dataset, present_algorithms, name))

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
