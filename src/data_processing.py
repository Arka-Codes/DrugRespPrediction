import pandas as pd
import numpy as np
from typing import cast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

class DrugResponseLoader:
    """
    Handles data loading, matching, and basic cleaning for GDSC drug response analysis.
    """
    
    def __init__(self, expr_path, resp_path):
        self.expr_path = expr_path
        self.resp_path = resp_path
        self.raw_expr = None
        self.raw_resp = None
        self.merged_data = None
        
    def load_data(self):
        """Loads the raw CSV files."""
        print("Loading datasets...")
        self.raw_expr = pd.read_csv(self.expr_path)
        self.raw_resp = pd.read_csv(self.resp_path)
        
        # Initial cleanup for Expression data
        # GDSC often has 'GENE_SYMBOLS' as a column. We want it as index.
        if 'GENE_SYMBOLS' in self.raw_expr.columns:
            self.raw_expr = self.raw_expr.set_index('GENE_SYMBOLS')
        
        # Transpose expression matrix so: distinct SAMPLES (Cell Lines) are ROWS, GENES are COLUMNS
        # This is the standard ML format (X = n_samples x n_features)
        # The raw file usually has COSMIC_IDs as columns.
        self.raw_expr = self.raw_expr.T
        
        # Clean up index names (remove 'COSMIC_' prefix to match integer IDs in response df if necessary)
        # Our mock generation added 'COSMIC_' prefix. Real GDSC often has it too or mixing.
        # Let's standardize to integer indices.
        self.raw_expr.index = self.raw_expr.index.str.replace('COSMIC_', '').astype(int)
        print(f"Expression Data Loaded: {self.raw_expr.shape} (Samples x Genes)")
        
    def filter_drug(self, drug_name):
        """Filters the response dataset for a specific drug."""
        if self.raw_resp is None:
            raise ValueError("Data not loaded. Call load_data() first before filtering.")
        print(f"Filtering response data for drug: {drug_name}")
        self.raw_resp = self.raw_resp[self.raw_resp['DRUG_NAME'] == drug_name]
        
        # Check for duplicates or multiple assays per cell line (ignoring for simple mock)
        # In real analysis, you might average them.
        self.raw_resp = self.raw_resp.groupby('COSMIC_ID')['LN_IC50'].mean().reset_index()
        self.raw_resp = self.raw_resp.dropna(subset=['LN_IC50'])
        print(f"Cell lines with valid IC50 for {drug_name}: {self.raw_resp.shape[0]}")
        
    def match_cell_lines(self):
        """
        Intersects the Cell Lines found in Expression matrix and Drug Response table.
        """
        if self.raw_expr is None or self.raw_resp is None:
            raise ValueError("Data not loaded. Call load_data() first before matching cell lines.")
        
        # Get common IDs
        expr_ids = set(self.raw_expr.index)
        resp_ids = set(self.raw_resp['COSMIC_ID'])
        common_ids = sorted(list(expr_ids.intersection(resp_ids)))
        
        if len(common_ids) == 0:
            raise ValueError("No common cell lines found between datasets! Check IDs format.")
            
        print(f"Matching cell lines... Found {len(common_ids)} common cell lines.")
        
        # Subset both
        X = self.raw_expr.loc[common_ids]
        y = self.raw_resp.set_index('COSMIC_ID').loc[common_ids]['LN_IC50']
        
        self.merged_data = (X, y)
        return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Applies the research-grade preprocessing pipeline:
    1. Train-Test Split (Prevent leakage!)
    2. Low Variance Filtering (on Train, applied to Test)
    3. Normalization (Z-score)
    """
    print("\n--- Starting Preprocessing ---")
    
    # 1. Train-Test Split
    # CRITICAL: Split BEFORE scaling or selection to avoid information leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 2. Low Variance Filtering
    # Remove genes that don't change across samples (noise/constant)
    # Threshold 0.5 is arbitrary but common for log-expression arrays.
    selector = VarianceThreshold(threshold=0.5) 
    
    # Fit on TRAIN only
    X_train_transformed = selector.fit_transform(X_train)
    # Transform TEST
    X_test_transformed = selector.transform(X_test)
    
    # Convert to dense numpy arrays
    from scipy import sparse
    from scipy.sparse import csr_matrix
    X_train_var: np.ndarray
    X_test_var: np.ndarray
    if sparse.issparse(X_train_transformed):
        X_train_var = cast(csr_matrix, X_train_transformed).toarray()
    else:
        X_train_var = np.asarray(X_train_transformed)
    if sparse.issparse(X_test_transformed):
        X_test_var = cast(csr_matrix, X_test_transformed).toarray()
    else:
        X_test_var = np.asarray(X_test_transformed)
    
    # Get surviving features
    genes_kept = X.columns[selector.get_support()]
    print(f"Variance Filtering: Dropped {X.shape[1] - len(genes_kept)} genes. Remaining: {len(genes_kept)}")
    
    # Convert back to DataFrame for readability
    X_train = pd.DataFrame(X_train_var, index=X_train.index, columns=genes_kept)
    X_test = pd.DataFrame(X_test_var, index=X_test.index, columns=genes_kept)
    
    # 3. Z-Score Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=genes_kept)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=genes_kept)
    
    print("Preprocessing Complete.")
    return X_train, X_test, y_train, y_test
