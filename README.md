 # isolation_forest-aad
Comparison of different anomaly detection algorithms based on Isolation Forest for interactive anomaly detection.

- The purpose was to rapidly prototype new algorithms and compare them. *These prototype implementations should be optimized before using them in other scenarios.* The algorithms also do not support to set the parameters for Isolation Forest (number of trees...).
- Algorithms using TiWS-iForest are based on code from https://github.com/tombarba/TinyWeaklyIsolationForest (weakly_supervised.ipynb, MIT License, Copyright (c) 2021 tombarba).
- IF-AAD and OMD are implemented by scripts that execute the original implementations, which need to be cloned into the same parent directory as this repository:
  - IF-AAD: https://github.com/shubhomoydas/ad_examples
  - OMD: https://github.com/siddiqmd/FeedbackIsolationForest

```pip install -r requirements.txt```

## Algorithms
Each subdirectory provides a single algorithm.

Algorithms can be executed running the python script `run-detection.py -a ALGORITHMS -ds DATASETS -r RUNS`

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

Script `prepare_datasets.py` is used to collect information about the datasets, which is stored in info.json.

## Results
The results directory contains numerical results of each run of the anomaly detection algorithms.

Folder structure: results/DATASET/ALGORITHM

### Original Results
* queried_instances-DATA_FILE#RUN.csv: Each line contains the index of the queried item (typically the most anomalous instance).

### Calculated Metrics
Using the results, following metrics are calculated:
* anomalies_seen-DATA_FILE#RUN.csv: Each line contains the number of anomalies seen of the run.
* anomalies_seen-DATA_FILE.csv: Each line contains the number of anomalies seen averaged over all runs on the data file and the values for the confidence interval.
* anomalies_seen.csv: Each line contains the number of anomalies seen averaged over all runs on all data files and the values for the values for the confidence interval.
* precision.csv: Each line contains the average precision@n


Metrics are calculated using the script `calc_metrics.py [DATASETS] [ALGORITHMS]`

## Plots
Following plots are generated from the metrics (one plot for each data set,
showing all algorithms):
* Anomalies / iteration
* Trained trees / iteration (only for applicable algorithms)

Plots are generated using the script `create_plots.py [DATASETS] [ALGORITHMS]`

