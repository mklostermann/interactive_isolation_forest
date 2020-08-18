import argparse
import helper
import algorithms.iforest_aad.iforest_aad
import algorithms.iforest.iforest

def main():
    parser = argparse.ArgumentParser(description="Runs anomaly detection.")
    parser.add_argument("-a", "--algorithms", type=str, nargs="*",
                        help="algorithms to run the detection on (all if omitted)")
    parser.add_argument("-ds", "--datasets", type=str, nargs="*",
                        help="data sets to run the detection on (all if omitted)")
    parser.add_argument("-b, --budget", dest="budget",  default=35, type=int, help="budget for feedback (default=35)")
    parser.add_argument("-r, --runs", dest="runs", default=1, type=int, help="number of repeated runs (default=1)")

    args = parser.parse_args()

    if args.algorithms is None:
        args.algorithms = helper.get_all_algorithms()

    if args.datasets is None:
        args.datasets = helper.get_all_datasets()

    for algorithm in args.algorithms:
        if algorithm == "iforest_aad":
            algorithms.iforest_aad.iforest_aad.detect(args.datasets, args.budget, args.runs)
        elif algorithm == "iforest":
            algorithms.iforest.iforest.detect(args.datasets, args.budget, args.runs)


if __name__ == '__main__':
    main()
