import os
from sklearn.ensemble import IsolationForest
import numpy as np
import helper

# Isolation Forest from scikit-learn
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "iforest")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        data, labels = helper.load_dataset(data_file)

        for run in range(1, runs + 1):
            iforest = IsolationForest(n_estimators=100, max_samples=256).fit(data)

            # Inversion required, as it returns the "opposite of the anomaly score defined in the original paper"
            scores = -iforest.score_samples(data)

            queried_instances = np.argsort(-scores)[np.arange(actual_budget)]
            helper.save_queried_instances(queried_instances, results_dir, data_file, run)
