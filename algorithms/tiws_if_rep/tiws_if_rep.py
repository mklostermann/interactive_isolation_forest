import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length

import helper

# TiWS-iForest applied to IAD, repeatedly trained (creates TiWS-iForest in each iteration, using a new IF)
# Implementation uses original code from https://github.com/tombarba/TinyWeaklyIsolationForest (weakly_supervised.ipynb,
# MIT license, Copyright (c) 2021 tombarba).
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "tiws_if_rep")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Read data sets
        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        data, labels = helper.load_dataset(data_file)

        for run in range(1, runs + 1):
            queried_instances = []
            # Calculate additional metrics (initialized with 0 as plots start at iteration 0)
            active_trees = [0]
            trained_trees = [0]

            for i in range(0, actual_budget):
                forest = IsolationForest(n_estimators=100, max_samples=256).fit(data)
                trained_trees.append(100)

                if i == 0:
                    # Initially, we only have a plain isolation forest; inversion of score required, as it returns the
                    # "opposite of the anomaly score defined in the original paper"
                    active_trees.append(100)
                    scores = -forest.score_samples(data)
                    queried = np.argsort(-scores)[0]
                else:
                    # Use TiWS-iForest with increasing feedback
                    supervised_data = data[queried_instances]
                    supervised_labels = labels[queried_instances]

                    # OPTIMIZE FOREST (using TiWS-iForest)
                    # Tree anomaly scores on supervised data
                    _, tree_supervised = compute_tree_anomaly_scores(forest, supervised_data)

                    # Average precision of trees on supervised data
                    ap_tree_supervised = np.array(
                        [measure(supervised_labels, - __tree_supervised__) for __tree_supervised__ in
                         tree_supervised])  ## MINUS SIGN

                    # learned_ordering sorts trees according to there performance. This will be used to create the
                    # candidate forests for TiWS-iForest.
                    learned_ordering = np.argsort(ap_tree_supervised)[::-1]
                    sorted_tree_supervised = tree_supervised[learned_ordering]

                    # Average precision of each candidate forest on supervised data
                    ap_forest_supervised = get_multiple_forest_average_precision(sorted_tree_supervised,
                                                                                 supervised_labels)

                    # The number of trees that were in the best forest
                    n_trees = get_last_occurrence_argmax(ap_forest_supervised) + 1
                    active_trees.append(n_trees)
                    # Store the TiWS-iForest in the existing forest variable (note: simple but inefficient).
                    tiws_indices = learned_ordering[0:n_trees]
                    forest.estimators_ = list(np.array(forest.estimators_)[tiws_indices])
                    forest.estimators_features_ = list(np.array(forest.estimators_features_)[tiws_indices])
                    forest.n_estimators = n_trees

                    # Evaluate the forest
                    # Note: This is very inefficient, all the trees are evaluated twice (now and during TiWS optimize)!
                    scores = -forest.score_samples(data)
                    for j in range(0, dataset_info.samples_count):
                        queried = np.argsort(-scores)[j]
                        if queried not in queried_instances:
                            break

                queried_instances.append(queried)

            helper.save_queried_instances(queried_instances, results_dir, data_file, run)
            helper.save_active_trees(active_trees, results_dir, data_file, run)
            helper.save_trained_trees(trained_trees, results_dir, data_file, run)


# Unmodified original source from weakly_supervised.ipynb
def get_multiple_forest_average_precision(tree, labels):
    # tree      : collection of anomaly scores for each tree
    # forest    : collection of anomaly scores for each forest, built with the 1st tree, 1st 2 trees, 1st 3 trees, and so on..
    # ap_forest : collection of the average precision of each single forest

    # rolling mean
    forest = (tree.cumsum(axis=0).T / np.arange(1, tree.shape[
        0] + 1)).T

    # average precision of each forest
    ap_forest = np.array([measure(labels, - __forest__) for __forest__ in forest])  ## MINUS SIGN

    return ap_forest

# Unmodified original source from weakly_supervised.ipynb
def compute_tree_anomaly_scores(forest, X):
    """
    Compute the score of each samples in X going through the extra trees.
    Parameters
    ----------
    X : array-like or sparse matrix
        Data matrix.
    subsample_features : bool
        Whether features should be subsampled.
    """
    n_samples = X.shape[0]

    depths = np.zeros(n_samples, order="f")

    collection_tree_anomaly_scores = []

    # print(len(forest.estimators_))

    for tree in forest.estimators_:
        leaves_index = tree.apply(X)
        node_indicator = tree.decision_path(X)
        n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

        tree_anomaly_scores = (
            np.ravel(node_indicator.sum(axis=1))
            + _average_path_length(n_samples_leaf)
            - 1.0)

        depths += tree_anomaly_scores

        collection_tree_anomaly_scores.append(tree_anomaly_scores)
        # print(forest.estimators_.index(tree))

        denominator = len(forest.estimators_) * _average_path_length([forest.max_samples_])
        scores = 2 ** (
            # For a single training sample, denominator and depth are 0.
            # Therefore, we set the score manually to 1.
            -np.divide(
                depths, denominator, out=np.ones_like(depths), where=denominator != 0
            )
        )

    collection_tree_anomaly_scores = np.array(collection_tree_anomaly_scores)

    return scores, collection_tree_anomaly_scores


# Slightly modified, original source from weakly_supervised.ipynb
def measure(y_true, y_pred):
    average_precision_score = metrics.average_precision_score(y_true, y_pred)

    return average_precision_score


# Unmodified original source from weakly_supervised.ipynb
def get_last_occurrence_argmax(x):
    argmax = x[::-1].argmax()
    argmax = x.shape[0] - argmax - 1

    return argmax
