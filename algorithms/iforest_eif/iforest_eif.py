import os
import eif
import numpy as np
import helper


# Isolation Forest as implemented by the eif (Extended Isolation Forest) module
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "iforest_eif")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        data, labels = helper.load_dataset(data_file)
        samples = min(dataset_info.samples_count, 256)

        for run in range(1, runs + 1):
            iforest = eif.iForest(data, ntrees=100, sample_size=samples, ExtensionLevel=0)
            scores = iforest.compute_paths(data)

            all_scores = np.vstack([scores] * (actual_budget + 1))
            helper.save_all_scores(all_scores, results_dir, data_file, run)

            queried_instances = np.argsort(-scores)[np.arange(actual_budget)]
            helper.save_queried_instances(queried_instances, results_dir, data_file, run)
