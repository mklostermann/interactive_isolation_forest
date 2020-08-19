import os
import eif
import numpy as np
import helper

# Extended Isolation Forest from eif
def detect(datasets, budget, runs):
    for dataset in datasets:
        data_file = helper.get_data_file(dataset)
        data, labels = helper.load_dataset(data_file)

        results_dir = helper.get_results_dir(dataset, "extended_iforest")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for run in range(1, runs + 1):
            iforest = eif.iForest(data, ntrees=100, sample_size=256, ExtensionLevel=data.shape[1] - 1)
            scores = iforest.compute_paths(data)

            all_scores = np.vstack([scores] * (budget + 1))
            all_scores_file = os.path.join(results_dir, f"all_scores-{run}.csv")
            np.savetxt(all_scores_file, all_scores, fmt='%f', delimiter=',')

            queried_instances = np.argsort(-scores)[np.arange(budget)]
            queried_file = os.path.join(results_dir, f"queried_instances-{run}.csv")
            np.savetxt(queried_file, queried_instances, fmt="%d", delimiter=",")
