import argparse
import helper
import algorithms.iforest.iforest
import algorithms.iforest_rep.iforest_rep
import algorithms.iforest_rep_dp.iforest_rep_dp
import algorithms.tiws_if.tiws_if
import algorithms.iif.iif
import algorithms.iforest_aad.iforest_aad
import algorithms.tiws_if_replace1.tiws_if_replace1
import algorithms.tiws_if_replace2.tiws_if_replace2
import algorithms.omd.omd
import threading
import time


class DetectorThread(threading.Thread):
    def __init__(self, algorithm, args):
        threading.Thread.__init__(self)
        self.algorithm = algorithm
        self.args = args

    def run(self):
        if self.algorithm == "iforest":
            algorithms.iforest.iforest.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "iforest_rep":
            algorithms.iforest_rep.iforest_rep.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "iforest_rep_dp":
            algorithms.iforest_rep_dp.iforest_rep_dp.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "tiws_if":
            algorithms.tiws_if.tiws_if.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "tiws_if_replace1":
            algorithms.tiws_if_replace1.tiws_if_replace1.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "tiws_if_replace2":
            algorithms.tiws_if_replace2.tiws_if_replace2.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "iif":
            algorithms.iif.iif.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "iforest_aad":
            algorithms.iforest_aad.iforest_aad.detect(self.args.datasets, self.args.budget, self.args.runs)
        elif self.algorithm == "omd":
            algorithms.omd.omd.detect(self.args.datasets, self.args.budget, self.args.runs)
        else:
            print(f"Unknown algorithm {self.algorithm} is ignored.")


def main():
    parser = argparse.ArgumentParser(description="Runs anomaly detection.")
    parser.add_argument("-a", "--algorithms", type=str, nargs="*",
                        help="algorithms to run the detection on (all if omitted)")
    parser.add_argument("-ds", "--datasets", type=str, nargs="*",
                        help="data sets to run the detection on (all if omitted)")
    parser.add_argument("-norm", "--normalized", type=bool,
                        help="Only normalized datasets")
    parser.add_argument("-withoutdup", "--without_duplicates", type=bool,
                        help="Only datasets without duplicates")
    parser.add_argument("-b, --budget", dest="budget", default=-1, type=int,
                        help="budget for feedback (default=auto=|O|)")
    parser.add_argument("-r, --runs", dest="runs", default=1, type=int, help="number of repeated runs (default=1)")
    parser.add_argument("-t, --threads", dest="threads", default=4, type=int,
                        help="number of threads used to start algorithms in parallel (default=4)")

    args = parser.parse_args()

    if args.algorithms is None:
        args.algorithms = helper.get_all_algorithms()

    dataset_info = helper.get_dataset_info()

    args.datasets = [info for info in dataset_info
                     if (args.datasets is None or info.dataset in args.datasets)
                     and (args.normalized is None or info.normalized)
                     and (args.without_duplicates is None or info.without_duplicates)]

    print(f"Running {args.algorithms} on {len(args.datasets)} datasets")
    print(f"Dataset files:")
    for info in args.datasets:
        print(info.get_data_file())

    threads = []
    for algorithm in args.algorithms:
        while True:
            if threading.active_count() - 1 < args.threads:
                thread = DetectorThread(algorithm, args)
                thread.start()
                threads.append(thread)
                break
            else:
                time.sleep(100)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
