import subprocess
import os

import helper

# OMD (Feedback Isolation Forest) by Siddiqui et al.
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "omd")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
# TODO: set learning rate (...) as in paper
        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        omd_args = f"./iforest.exe -i {data_file} -o {results_dir}/omd -t 100 -s 256 -m 1 -l 2 -a 1 -w 2 -f {actual_budget} -x {runs}"
        subprocess.run(omd_args.split(" "), cwd=os.path.abspath("../FeedbackIsolationForest"))

        # Prepare original output files for metrics calculation by removing files we do not need.
        for file in os.scandir(results_dir):
            # Delete: Found anomalies for original IF without feedback, cost during optimization
            if file.name.__contains__("_summary_feed_0_") or file.name.__contains__("_cost_feed_"):
                os.remove(file)
