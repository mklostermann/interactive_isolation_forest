import os
import numpy as np
from sklearn.ensemble import IsolationForest

import helper

# Unsupervised IF, repeatedly trained, with data pruning (train trees only on unlabeled and anomalous data).
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "iforest_rep_dp")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        data, labels = helper.load_dataset(data_file)

        for run in range(1, runs + 1):
            queried_instances = []

            for i in range(0, actual_budget):
                pruned_data_indices = [i for i in range(dataset_info.samples_count)
                                       if (i not in queried_instances or labels[i] == 0)]
                iforest = IsolationForest(n_estimators=100, max_samples=256).fit(data[pruned_data_indices])

                # Inversion required, as it returns the "opposite of the anomaly score defined in the original paper"
                scores = -iforest.score_samples(data)
                for j in range(0, dataset_info.samples_count):
                    queried = np.argsort(-scores)[j]
                    if queried not in queried_instances:
                        break

                queried_instances.append(queried)

            helper.save_queried_instances(queried_instances, results_dir, data_file, run)