import os
import sklearn as sk
import sklearn.ensemble
import numpy as np
import helper

# Isolation Forest from scikit-learn
def detect(datasets, budget, runs):
    for dataset in datasets:
        results_dir = helper.get_results_dir(dataset, "iforest_sklearn")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for data_file in helper.get_all_data_files(dataset):
            data, labels = helper.load_dataset(data_file)

            for run in range(1, runs + 1):
                iforest = sk.ensemble.IsolationForest(n_estimators=100, max_samples=256).fit(data)

                # Inversion required, as it returns the "opposite of the anomaly score defined in the original paper"
                scores = -iforest.score_samples(data)

                all_scores = np.vstack([scores] * (budget + 1))
                helper.save_all_scores(all_scores, results_dir, data_file, run)

                queried_instances = np.argsort(-scores)[np.arange(budget)]
                helper.save_queried_instances(queried_instances, results_dir, data_file, run)
