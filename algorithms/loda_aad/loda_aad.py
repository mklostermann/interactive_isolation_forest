import subprocess
import tempfile
import os

import helper


def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "loda_aad")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_runs = runs if dataset_info.downsampled_with_variations else runs * 10
        model_file = tempfile.NamedTemporaryFile(suffix=".mdl").name
        aad_args = f"python3 -m ad_examples.aad.aad_batch --resultsdir={results_dir} --dataset={dataset_info.dataset} --datafile={data_file} --reruns={actual_runs} --budget={budget} --modelfile={model_file} --log_file=if_aad.log --startcol=2 --labelindex=1 --header --randseed=42 --querytype=1 --detector_type=13 --constrainttype=4 --sigma2=0.5 --runtype=multi --reps=1 --maxbudget=10000 --topK=0 --init=1 --tau=0.03 --forest_n_trees=100 --forest_n_samples=256 --forest_score_type=4 --forest_add_leaf_nodes_only --forest_max_depth=100 --tau_score_type=1 --Ca=1 --Cn=1 --Cx=1 --withprior --unifprior --norm_unit --mink=300 --maxk=500 --prior_influence=1 --max_anomalies_in_constraint_set=1000 --max_nominals_in_constraint_set=1000 --n_explore=10 --num_query_batch=1 --cachedir= --tree_update_type=0 --max_windows=30 --query_euclidean_dist_type=0 --min_feedback_per_window=2 --max_feedback_per_window=20 --allow_stream_update --stream_window=512 --retention_type=0 --till_budget --forest_replace_frac=0.2 --check_KL_divergence --kl_alpha=0.05 --n_pretrain=50 --n_pretrain_nominals=10 --n_weight_updates_after_stream_window=10 --rule_output_interval=20 --debug"
        subprocess.run(aad_args.split(" "), cwd=os.path.abspath("../ad_examples"))
