import os
import sklearn as sk
import sklearn.ensemble
import numpy as np
import helper

# Isolation Forest from scikit-learn
def detect(datasets, budget, runs):
    for dataset in datasets:
        data_file = helper.get_data_file(dataset)
        data, labels = helper.load_dataset(data_file)

        results_dir = helper.get_results_dir(dataset, "iforest_sklearn")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for run in range(1, runs + 1):
            iforest = sk.ensemble.IsolationForest(n_estimators=100, max_samples=256).fit(data)

            # Inversion required, as it returns the "opposite of the anomaly score defined in the original paper"
            scores = -iforest.score_samples(data)

            all_scores = np.vstack([scores] * (budget + 1))
            all_scores_file = os.path.join(results_dir, f"all_scores-{run}.csv")
            np.savetxt(all_scores_file, all_scores, fmt='%f', delimiter=',')

            queried_instances = np.argsort(-scores)[np.arange(budget)]
            queried_file = os.path.join(results_dir, f"queried_instances-{run}.csv")
            np.savetxt(queried_file, queried_instances, fmt="%d", delimiter=",")
