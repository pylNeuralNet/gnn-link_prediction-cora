# gnn-link_prediction-cora
```
PŁ/Sekcja AI SKN/GNN1/Exercises/
├── Exercise 1 - basic graph exploration and metrics
├── Exercise 2 (this repo) -> Publication datasets -> Cora (with timestamps)
└──
```

# To contribute
clone this repo

```bash
git branch --list # optional
git checkout -b yourBranch # for example git checkout -b dev/jaccard_coeff or prod/jaccard_coeff
# this is because afterwards when you're done with your changes, we can merge it to the main branch

# implement your contribution
# (from root)
git add -A # or each file that you want to stage separately
git status # optional - to verify if you added the right files
git commit # then edit commit message, in vsc a new window pops out. The title should be a concise slogan of the change, description should detail for example features, what they do and what for
git push origin yourBranch
# when you're done, create merge request
```

Add new classes to /src/metrics/
Call them in run_analysis.py similar to CommonNeighbors.

New datasets can be added to /src/data/data_loader.py and downloader.py, to config.yaml. 

# Project structure
```
src/
├── run_analysis.py
├── data/
│   └── __init__.py
│   └── base.py
│   └── data_loader.py
│   └── downloader.py
│   └── preprocessor.py
├── metrics/
│   └── __init__.py
│   └── base_metric.py
│   └── common_neighbors.py
│   └── utils.py
├── models/
│   └── __init__.py
│   └── graph.py
├── utils/
├── config/
│   └── __init__.py
│   └── config.yaml
└── tests/
    └── __init__.py
    └── test_metrics.py
```

# Use
## Venv
```bash
# From project root
# optional: replace venv with your custom virtual environment name
python -m venv .venv

# Activate (Unix)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```
Optional:
`python.exe -m pip install --upgrade pip`

## Installation
`pip install torch torch-geometric networkx pyyaml scikit-learn`
<!-- `pip freeze | grep -E "torch|^torch-|^torch=|^torch$|torch-geometric|pytorch" > requirements.txt` -->
`pip install -r requirements.txt`

## Run
```bash
cd src/
python -m run_analysis
```

# Analysis
- Common neighbors
- Jaccard Coeff
- Academic-Adar Index*
- Preferential Attachmen

## Common Neighbors
<!-- TODO: move to docs/ -->
### Sample output
```
2024-11-09 15:56:31,222 - data.downloader - INFO - Successfully loaded Cora dataset
2024-11-09 15:56:31,223 - __main__ - INFO - Dataset loaded: 2708 nodes, 10556 edges
2024-11-09 15:56:32,366 - __main__ - INFO - Generated 3660000 candidate pairs
2024-11-09 15:56:37,714 - __main__ - INFO - 
Top 100 predicted links:
2024-11-09 15:56:37,729 - __main__ - INFO - Node 1623 - Node 306: Score = 20
2024-11-09 15:56:37,730 - __main__ - INFO - Node 1701 - Node 598: Score = 12
2024-11-09 15:56:37,730 - __main__ - INFO - Node 1986 - Node 1701: Score = 11
2024-11-09 15:56:37,730 - __main__ - INFO - Node 507 - Node 1542: Score = 11
2024-11-09 15:56:37,732 - __main__ - INFO - Node 1483 - Node 2450: Score = 10
```

### Explanation
The score represents the number of common neighbors between two nodes
For example, node 1623 and node 306 have a score of 20, meaning they share 20 common neighbors despite not being directly connected
Higher scores suggest stronger likelihood of a connection as these nodes have more mutual connections