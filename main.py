import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Add global project path to sys.path to ensure modules can be imported
sys.path.append(os.getcwd())

from src.create_mock_data import create_mock_gdsc_data
from src.data_processing import DrugResponseLoader, preprocess_data
from src.models import ModelTrainer
from src.evaluation import plot_prediction_performance
from src.interpretation import ModelInterpreter

def main():
    parser = argparse.ArgumentParser(description="Run the Drug Response Prediction Pipeline")
    parser.add_argument("--drug", type=str, default="Cisplatin", help="Drug to predict response for (default: Cisplatin)")
    parser.add_argument("--generate-data", action="store_true", help="Force generation of mock data")
    args = parser.parse_args()

    # Paths
    DATA_DIR = "data"
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    RESULTS_DIR = "results"
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    EXPR_PATH = os.path.join(RAW_DIR, "mock_gene_expression.csv")
    RESP_PATH = os.path.join(RAW_DIR, "mock_drug_response.csv")

    # 1. Data Availability / Generation
    if args.generate_data or not os.path.exists(EXPR_PATH) or not os.path.exists(RESP_PATH):
        print("Required data not found or generation requested. Generating mock data...")
        create_mock_gdsc_data(output_dir=RAW_DIR)
    else:
        print("Using existing mock data found in data/raw/")

    # 2. Load and Preprocess Data
    print(f"\nLoading data for drug: {args.drug}...")
    loader = DrugResponseLoader(EXPR_PATH, RESP_PATH)
    loader.load_data()
    loader.filter_drug(args.drug)
    
    try:
        X, y = loader.match_cell_lines()
    except ValueError as e:
        print(f"Error matching cell lines: {e}")
        return

    # Preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 3. Model Training
    print("\nStarting Model Training...")
    trainer = ModelTrainer(X_train, y_train, X_test, y_test)
    
    # Train Baseline (Linear Regression)
    lr_model = trainer.train_baseline()
    
    # Train Random Forest
    rf_model = trainer.train_random_forest()
    
    # Train XGBoost
    xgb_model = trainer.train_xgboost()

    # 4. Evaluation and Visualization
    print("\nEvaluating Models...")
    models = {
        "LinearRegression": lr_model,
        "RandomForest": rf_model, 
        "XGBoost": xgb_model
    }

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        
        # Plotting
        plot_path = os.path.join(FIGURES_DIR, f"pred_vs_actual_{name}_{args.drug}.png")
        plot_prediction_performance(y_test, y_pred, model_name=name, save_path=plot_path)
        plt.savefig(plot_path)
        plt.close() # Close plot to free memory
        print(f"Saved performance plot to {plot_path}")

        # Interpretation (Feature Importance)
        if name in ["RandomForest", "XGBoost", "LinearRegression"]:
            interpreter = ModelInterpreter(model, X_train.columns)
            try:
                imp_df = interpreter.get_feature_importance()
                # Print top 10 genes
                print(f"Top 10 Genes for {name}:")
                print(imp_df.head(10))
                
                # Save importance plot
                imp_plot_path = os.path.join(FIGURES_DIR, f"feature_importance_{name}_{args.drug}.png")
                interpreter.plot_top_genes(top_n=20, save_path=imp_plot_path)
                # plt.savefig(imp_plot_path) # plot_top_genes already saves it
                plt.close()
                print(f"Saved feature importance plot to {imp_plot_path}")
            except Exception as e:
                print(f"Could not interpret model {name}: {e}")

    print("\nPipeline execution complete!")

if __name__ == "__main__":
    main()


#run command for entire pipeline (using drug: Cisplatin)
#export PYTHONPATH=$PYTHONPATH:$(pwd) && .venv/bin/python main.py --drug Cisplatin