# Music Cluster Lab

Streamlit application for clustering Spotify song data by numerical musical features. It helps discover similar songs with different clustering algorithms, compare metrics, and export results as CSV.

## Features

- Upload a CSV dataset
- Select musical features
- Scale data with `StandardScaler`, `MinMaxScaler`, or `MaxAbsScaler`
- K-Means, K-Medoids, DBSCAN ve Gaussian Mixture Model (GMM)
- Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics
- ARI, NMI, Homogeneity, Completeness, and V-measure metrics when genre data is available
- Cluster sizes, cluster profiles, and PCA visualization
- Similar song recommendations from the same cluster
- Download cluster results as CSV
- Cache measurements for the selected algorithm
- Rerun the selected algorithm with the `Measure again` button

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run app.py
```

When the application opens, upload a CSV file and select the features, scaling method, and algorithm from the sidebar.

## Dataset format

The CSV file must contain at least two numerical musical features. The main supported features are:

- `danceability`
- `energy`
- `tempo`
- `acousticness`
- `instrumentalness`
- `valence`
- `loudness`
- `speechiness`
- `liveness`

Metadata columns such as `track_name`, `artists`, and genre are optional. External validation metrics are calculated when a genre column is available.

To validate a dataset before running the application:

```bash
python validate_dataset.py data/dataset.csv
```

## Measurement cache

The application caches measurement results for the same data, features, scaling method, and algorithm parameters. When an algorithm is selected again, the model and metrics do not need to be recalculated.

To create a new measurement, click **Measure again** in the sidebar. This clears the cache and reruns only the currently selected algorithm.

## Project files

- `app.py`: Streamlit user interface and visualizations
- `clustering.py`: Clustering algorithms, metrics, and cache management
- `config.py`: Feature, metadata, and column mappings
- `metadata.py`: Song name and metadata helpers
- `validate_dataset.py`: CSV dataset validation tool
- `clean_tempo.py`: Tempo data cleaning tool
- `data/`: Example and processed datasets
