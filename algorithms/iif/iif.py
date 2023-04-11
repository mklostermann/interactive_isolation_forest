import os
import sklearn as sk
import sklearn.ensemble
import numpy as np
import helper

# Interactive Isolation Forest (new algorithm)
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "iif")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        data, labels = helper.load_dataset(data_file)

        for run in range(1, runs + 1):
            iforest = sk.ensemble.IsolationForest(n_estimators=100, max_samples=256).fit(data)

            # Inversion required, as it returns the "opposite of the anomaly score defined in the original paper"
            scores = -iforest.score_samples(data)

            all_scores = np.vstack([scores] * (actual_budget + 1))
            helper.save_all_scores(all_scores, results_dir, data_file, run)

            queried_instances = np.argsort(-scores)[np.arange(actual_budget)]
            helper.save_queried_instances(queried_instances, results_dir, data_file, run)
