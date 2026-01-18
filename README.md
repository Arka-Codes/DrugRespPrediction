# AI-Driven Drug Response Prediction (DRP)

## Project Overview

This project implements a reproducible machine learning pipeline to predict cancer drug sensitivity (IC50) of cell lines using high-dimensional gene expression profiles. It mimics a real-world bioinformatics workflow using data structures similar to the **Genomics of Drug Sensitivity in Cancer (GDSC)** database.

The pipeline covers the entire data science lifecycle: from data ingestion and cleaning to model training, evaluation, and biological interpretation of results.

## Key Features

- **Robust Preprocessing:** Handles missing data, matches cell lines across datasets, filters low-variance genes, and applies Z-score normalization.

- **Multi-Model Comparison:** Trains and benchmarks three distinct regression models:
  - **Linear Regression** (Baseline)
  - **Random Forest** (Ensemble Bagging)
  - **XGBoost** (Gradient Boosting)
- **Biological Interpretability:** Extracts and visualizes feature importance to identify which genes drive drug sensitivity predictions.
- **Reproducibility:** A unified `main.py` orchestrator ensures the entire experiment can be re-run with a single command.

## Project Structure

```mermaid
├── data/                   # Data storage
│   ├── raw/                # Raw CSV inputs (mock generated or real)
│   └── processed/          # Intermediate processed files (optional)
├── results/                # Experiment outputs
│   ├── figures/            # Performance & importance plots
│   └── metrics/            # Numerical results (CSV/JSON)
├── src/                    # Source code modules
│   ├── create_mock_data.py # Synthetic data generator
│   ├── data_processing.py  # Cleaning, scaling, and splitting logic
│   ├── models.py           # Model definitions (LR, RF, XGB)
│   ├── evaluation.py       # Plotting and metric calculation
│   └── interpretation.py   # Feature importance analysis
├── docs/                   # Documentation and data dictionaries
├── main.py                 # Main entry point for the pipeline
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd DRP
   ```

2. **Set up a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project is controlled via the `main.py` script. It can automatically generate mock data if local files are missing.

### Basic Run

Run the pipeline for the default drug (**Cisplatin**):

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)  # Ensure src module is visible
python main.py
```

### Command Line Arguments

- `--drug`: Specify the drug name to analyze (e.g., 'Cisplatin', 'Doxorubicin', 'Paclitaxel').
- `--generate-data`: Force regeneration of synthetic mock datasets.

### Examples

Predict response for **Doxorubicin**:

```bash
python main.py --drug Doxorubicin
```

Force fresh data generation and run for **Paclitaxel**:

```bash
python main.py --drug Paclitaxel --generate-data
```

## Pipeline Details

1. **Data Ingestion (`src/data_processing.py`):**
   - Loads Gene Expression (Features) and Drug Response (Targets).
   - Transposes expression matrices to standard `(n_samples, n_features)` format.
   - Matches cell lines by COSMIC ID intersection.

2. **Preprocessing:**
   - **Splitting:** 80/20 Train-Test split.
   - **Filtering:** `VarianceThreshold(0.5)` removes noise/constant genes.
   - **Scaling:** `StandardScaler` standardizes gene expression to mean=0, var=1.

3. **Modeling (`src/models.py`):**
   - Trains Linear Regression (baseline).
   - Trains Random Forest (captures non-linearities).
   - Trains XGBoost (optimized gradient boosting).

4. **Evaluation (`results/figures/`):**
   - Generates "Predicted vs Actual" scatter plots.
   - Calculates Performance Metrics (RMSE, R² - displayed in logs).

5. **Interpretation (`src/interpretation.py`):**
   - Extracts top feature importance scores.
   - Plots the top 20 genes most predictive of drug response.

## Results

After a successful run, check the `results/figures/` directory for visual insights:

- `pred_vs_actual_<Model>_<Drug>.png`: Visual assessment of model fit.
- `feature_importance_<Model>_<Drug>.png`: Biological markers identified by the model.
