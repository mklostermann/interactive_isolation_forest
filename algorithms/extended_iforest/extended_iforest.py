import os
import eif
import numpy as np
import helper

# Extended Isolation Forest from eif
def detect(datasets, budget, runs):
    for dataset in datasets:
        results_dir = helper.get_results_dir(dataset, "extended_iforest")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for data_file in helper.get_all_data_files(dataset):
            data, labels = helper.load_dataset(data_file)

            for run in range(1, runs + 1):
                iforest = eif.iForest(data, ntrees=100, sample_size=256, ExtensionLevel=data.shape[1] - 1)
                scores = iforest.compute_paths(data)

                all_scores = np.vstack([scores] * (budget + 1))
                helper.save_all_scores(all_scores, results_dir, data_file, run)

                queried_instances = np.argsort(-scores)[np.arange(budget)]
                helper.save_queried_instances(queried_instances, results_dir, data_file, run)
