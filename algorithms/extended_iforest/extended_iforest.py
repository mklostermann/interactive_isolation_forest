import os
import eif
import numpy as np
import helper

# Extended Isolation Forest from eif
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "extended_iforest")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_runs = runs if dataset_info.downsampled_with_variations else runs * 10
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        data, labels = helper.load_dataset(data_file)

        for run in range(1, actual_runs + 1):
            samples = min(dataset_info.samples_count, 256)
            iforest = eif.iForest(data, ntrees=100, sample_size=samples, ExtensionLevel=data.shape[1] - 1)
            scores = iforest.compute_paths(data)

            all_scores = np.vstack([scores] * (actual_budget + 1))
            helper.save_all_scores(all_scores, results_dir, data_file, run)

            queried_instances = np.argsort(-scores)[np.arange(actual_budget)]
            helper.save_queried_instances(queried_instances, results_dir, data_file, run)
