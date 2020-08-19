# isolation_forest-aad
Comparison of different anomaly detection algorithms based on Isolation Forest with active learning (active anomaly detection).

```pip install -r requirements.txt```

## Algorithms
Each sub directory provides a single algorithm.

Algorithms can be executed running the python script `run-detection.py -ds DATASETS -r RUNS`

## Data Sets
Data sets are provided as CSV. The first column contains the label (nominal/anomaly),
followed by one column for each feature:
```
label,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13
nominal,0,1,0,0,0,0,0,0.085106,0.042482,0.251116,0.003467,0.422915,0.414912
anomaly,0,1,0,0,0,0,0,0.255319,0.051489,0.298721,0.003467,0.422915,0.414912
```

Each data set is located in a separate directory, which might contain additional
data. See the respective README.md for details about the data set.

## Results
The results directory contains numerical results of each run of the anomaly detection algorithms.

Folder structure: results/DATASET/ALGORITHM

### Original Results
* queried_instances-#RUN.csv: Each line contains the index of the queried item (typically the most anomalous instance).
* all_scores-#RUN.csv: Each line contains the anomaly scores of the current model (first line = iteration 0 -> no feedback)

### Calculated Metrics
Using the results, following metrics are calculated:
* anomalies_seen-RUN.csv: Each line contains the number of anomalies seen of the run.
* anomalies_seen.csv: Each line contains the number of anomalies seen averaged over all runs and the values for the 95% confidence intervals.
* auroc-RUN.csv: Each line contains the AUC ROC of that run.
* auroc.csv: Each line contains the AUC ROC averaged over all runs and the values for the 95% confidence intervals.


Metrics are calculated using the script `calc_metrics.py [DATASETS] [ALGORITHMS]`

## Plots
Following plots are generated from the metrics (one plot for each data set,
showing all algorithms):
* # Anomalies / Iteration
* AUC ROC / Iteration
* TODO?: P@n, R@n, F1@n

Plots are generated using the script `create_plots.py [DATASETS] [ALGORITHMS]`

