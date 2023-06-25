import copy
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import metrics
from sklearn.ensemble._iforest import _average_path_length
import matplotlib.pylab as plt

import helper


# TiWS-iForest applied to IAD with replacement of the removed trees AFTER evaluation.
# Implementation uses original code from https://github.com/tombarba/TinyWeaklyIsolationForest (weakly_supervised.ipynb)
def detect(datasets, budget, runs):
    for dataset_info in datasets:
        results_dir = helper.get_results_dir(dataset_info.dataset, "tiws_if_replace2")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        data_file = dataset_info.get_data_file()
        actual_budget = dataset_info.outlier_count if budget <= 0 else budget
        data, labels = helper.load_dataset(data_file)

        for run in range(1, runs + 1):
            queried_instances = []
            active_trees = [np.nan] # Plots start at iteration 0
            trained_trees = [np.nan]

            # Following code is mainly from weakly_supervised_algo()
            # It is important to train the initial unsupervised IF only once to see if there is an actual improvement.

            # UNSUPERVISED TRAIN --------------------------------------------------------------

            # unsupervised train on the full train_data
            # the weakly supervised train will be performed only on supervised_data
            # sk_IF is the standard sklearn Isolation Forest
            sk_IF = IsolationForest(n_estimators=100, max_samples=256).fit(data)
            trained_trees.append(100)
            # Initialize a second forest, which will be used to evaluate the TiWS-iForest.
            # Not very elegant, but an easy way to evaluate the forest later on.
            tiws_IF = copy.deepcopy(sk_IF)
            for i in range(0, actual_budget):

                if i == 0:
                    # Initially, we only have a plain isolation forest; inversion required, as it returns the "opposite
                    # of the anomaly score defined in the original paper"
                    active_trees.append(sk_IF.n_estimators)
                    scores = -sk_IF.score_samples(data)
                    queried = np.argsort(-scores)[0]
                else:
                    # MKL: Tree replacement
                    n_replaced_trees = sk_IF.n_estimators - tiws_IF.n_estimators
                    trained_trees.append(n_replaced_trees)
                    if n_replaced_trees > 0:
                        new_IF = IsolationForest(n_estimators=n_replaced_trees, max_samples=256).fit(data)
                        tiws_IF.estimators_.extend(new_IF.estimators_)
                        tiws_IF.estimators_features_.extend(new_IF.estimators_features_)
                        tiws_IF.n_estimators = sk_IF.n_estimators

                    # Use TiWS-iForest with increasing feedback
                    supervised_data = data[queried_instances]
                    supervised_labels = labels[queried_instances]

                    # WEAKLY SUPERVISED TRAIN --------------------------------------------------------------

                    # tree_supervised        : collection of anomaly scores for each tree in the forest, on supervised dataset
                    # ap_tree_supervised     : collection of the average precision of each tree in the forest, on supervised dataset
                    # learned_ordering       : order of the trees, from the best to the worst
                    # sorted_tree_supervised : collection of anomaly scores on the supervised data, sorted by learned_ordering
                    # ap_forest_supervised   : collection of the average precision of each tree in the forests obtained with the learned ordering

                    ## Test:
                    # tree_test              : collection of anomaly scores for each tree in the forest, on test dataset
                    # sorted_tree_test       : tree_test sorted according to learned_ordering
                    # ap_forest_test         : collection of average precision of each tree in the forests obtained with the learned ordering

                    # train anomaly scores
                    _, tree_supervised = compute_tree_anomaly_scores(tiws_IF,
                                                                     supervised_data)

                    # average precision of trees on the supervised dataset
                    ap_tree_supervised = np.array(
                        [measure(supervised_labels, - __tree_supervised__) for __tree_supervised__ in
                         tree_supervised])  ## MINUS SIGN

                    #  learn tree-order from supervised dataset
                    # MKL: learned_ordering sorts trees according to there performance. This will be used to create the
                    # candidate forests for TiWS-iForest.
                    learned_ordering = np.argsort(ap_tree_supervised)[::-1]
                    sorted_tree_supervised = tree_supervised[learned_ordering]

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

                    # compute average precision on train
                    # MKL: Use the sorted trees to find the best performing forest (= TiWS-iForest)
                    ap_forest_supervised = get_multiple_forest_average_precision(sorted_tree_supervised,
                                                                                 supervised_labels)

                    # MKL: The rest of the code is new:
                    # - Get the index of the TiWS-iForest and (from get_values())
                    # - Create the TiWS-iForest by using the trees for a new forest
                    # - Evaluate scores and query anomaly
                    n_trees = get_last_occurrence_argmax(ap_forest_supervised) + 1
                    active_trees.append(n_trees)
                    tiws_indices = learned_ordering[0:n_trees]
                    tiws_IF.estimators_ = list(np.array(tiws_IF.estimators_)[tiws_indices])
                    tiws_IF.estimators_features_ = list(np.array(tiws_IF.estimators_features_)[tiws_indices])
                    tiws_IF.n_estimators = n_trees

                    scores = -tiws_IF.score_samples(data)
                    for j in range(0, dataset_info.samples_count):
                        queried = np.argsort(-scores)[j]
                        if queried not in queried_instances:
                            break

                queried_instances.append(queried)

            helper.save_queried_instances(queried_instances, results_dir, data_file, run)
            helper.save_active_trees(active_trees, results_dir, data_file, run)
            helper.save_trained_trees(trained_trees, results_dir, data_file, run)


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


# Unmodified original source from weakly_supervised.ipynb
def measure(y_true, y_pred, plot=False):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    average_precision_score = metrics.average_precision_score(y_true, y_pred)

    if plot == True:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[5 * 3, 5])

        def ax_plot(ax, x, y, xlabel, ylabel, title=''):
            ax.plot(x, y);
            ax.set_xlabel(xlabel),;
            ax.set_ylabel(ylabel)
            ax.set_title(title);
            ax.grid()

        ax_plot(ax1, fpr, tpr, 'fpr', 'tpr', title="{:.3f}".format(auc))
        ax_plot(ax2, recall, precision, 'recall', 'precision', title="{:.3f}".format(average_precision_score))

    else:

        return average_precision_score


# Unmodified original source from weakly_supervised.ipynb
def get_last_occurrence_argmax(x):

    argmax  = x[::-1].argmax()
    argmax  = x.shape[0] - argmax - 1

    return argmax


# Unmodified original source from weakly_supervised.ipynb
def get_values(ap_forest_supervised, ap_forest_test):

    argmax_supervised = get_last_occurrence_argmax(ap_forest_supervised)
    argmax_test = get_last_occurrence_argmax(ap_forest_test)

    max_supervised = ap_forest_supervised[argmax_supervised]
    max_test = ap_forest_test[argmax_test]
    test_on_argmax_supervised = ap_forest_test[argmax_supervised]

    last_supervised = ap_forest_supervised[-1]
    last_test = ap_forest_test[-1]

    first_supervised = ap_forest_supervised[0]
    first_test = ap_forest_test[0]

    return {'argmax_supervised': argmax_supervised,
            'argmax_test': argmax_test,
            'max_supervised': max_supervised,
            'max_test': max_test,
            'test_on_argmax_supervised': test_on_argmax_supervised,
            'last_supervised': last_supervised,
            'last_test': last_test,
            'first_supervised': first_supervised,
            'first_test': first_test}
