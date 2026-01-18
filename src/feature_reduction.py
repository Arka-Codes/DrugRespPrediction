import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class FeatureReducer:
    """
    Handles dimensionality reduction using PCA.
    """
    def __init__(self, n_components=None):
        """
        n_components: Number of components to keep. If float (0 < n < 1), 
        selects num components to explain that fraction of variance.
        """
        self.pca = PCA(n_components=n_components)
        self.n_components_ = None
        
    def fit_transform(self, X_train):
        """
        Fits PCA on training data and returns transformed data.
        """
        print(f"Fitting PCA on train set with shape {X_train.shape}...")
        X_pca = self.pca.fit_transform(X_train)
        self.n_components_ = self.pca.n_components_
        print(f"PCA reduced to {self.n_components_} components.")
        
        # DataFrame wrapper
        cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, index=X_train.index, columns=cols)
    
    def transform(self, X_test):
        """
        Applies learned PCA projection to test data.
        """
        X_pca = self.pca.transform(X_test)
        cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, index=X_test.index, columns=cols)
    
    def plot_explained_variance(self, output_path=None):
        """
        Plots the cumulative explained variance ratio.
        """
        cum_var = np.cumsum(self.pca.explained_variance_ratio_)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        # Add threshold line for 90%
        plt.axhline(y=0.90, color='r', linestyle=':', label='90% Variance')
        plt.legend()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

def run_pca_analysis(X_train_path, X_test_path, output_dir='data/processed'):
    """
    Loads processed data, runs PCA, saves results and plot.
    """
    print("Loading processed data for PCA...")
    X_train = pd.read_csv(X_train_path, index_col=0)
    X_test = pd.read_csv(X_test_path, index_col=0)
    
    # Run PCA targeting 95% variance (common heuristic for omics)
    reducer = FeatureReducer(n_components=0.95)
    
    X_train_pca = reducer.fit_transform(X_train)
    X_test_pca = reducer.transform(X_test)
    
    # Save transformed data
    X_train_pca.to_csv(f"{output_dir}/X_train_pca.csv")
    X_test_pca.to_csv(f"{output_dir}/X_test_pca.csv")
    print("Saved PCA-transformed datasets.")
    
    return reducer
