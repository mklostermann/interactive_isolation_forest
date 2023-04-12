import subprocess
import tempfile
import os

import helper

# IF-AAD by Das et al. using parameters as suggested in paper (tree-based AAD).
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "iforest_aad")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        model_file = tempfile.NamedTemporaryFile(suffix=".mdl").name
        aad_args = f"python3 -m ad_examples.aad.aad_batch --resultsdir={results_dir} --dataset={dataset_info.dataset} --datafile={data_file} --reruns={runs} --budget={actual_budget} --modelfile={model_file} --log_file=if_aad.log --startcol=2 --labelindex=1 --header --randseed=42 --querytype=1 --detector_type=7 --constrainttype=4 --sigma2=0.5 --runtype=multi --reps=1 --init=1 --tau=0.03 --forest_n_trees=100 --forest_n_samples=256 --forest_score_type=4 --forest_max_depth=100 --tau_score_type=1 --Ca=100 --Cn=1 --Cx=0.001 --num_query_batch=1 --cachedir= --debug"
        subprocess.run(aad_args.split(" "), cwd=os.path.abspath("../ad_examples"))
