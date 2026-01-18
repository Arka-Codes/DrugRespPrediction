import pandas as pd
import numpy as np
import os

def create_mock_gdsc_data(output_dir='data/raw', n_cell_lines=200, n_genes=1000, n_drugs=5):
    """
    Generates synthetic gene expression and drug response data 
    mimicking the GDSC format for educational/testing purposes.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Generate Cell Line IDs (COSMIC_IDs)
    # Create some overlap, but also some unique to each dataset to simulate real-world mismatch
    common_ids = np.arange(1000, 1000 + n_cell_lines)
    expr_ids = np.concatenate([common_ids, np.arange(2000, 2020)]) # 20 extra in expression
    resp_ids = np.concatenate([common_ids, np.arange(3000, 3020)]) # 20 extra in response
    
    # 2. Mock Gene Expression Data (RMA normalized, typically 4-14 range)
    # Structure: Genes as rows, Cell Lines as columns (Standard Bioinfo format) or vice versa.
    # GDSC usually provides Genes as Rows. Let's stick to that, then transpose in processing.
    print(f"Generating mock expression data for {len(expr_ids)} cell lines and {n_genes} genes...")
    
    gene_symbols = [f"GENE_{i}" for i in range(1, n_genes + 1)]
    
    # Generate random expression data
    # Matrix shape: (n_genes, n_cell_lines)
    expr_matrix = np.random.normal(loc=7, scale=2, size=(n_genes, len(expr_ids)))
    
    # Add some low variance genes (to be filtered later)
    expr_matrix[0:10, :] = 10.0 # Constant genes
    
    df_expr = pd.DataFrame(expr_matrix, index=gene_symbols, columns=[f"COSMIC_{i}" for i in expr_ids])
    df_expr.index.name = 'GENE_SYMBOLS'
    df_expr.reset_index(inplace=True)
    
    # Save Expression Data
    expr_path = os.path.join(output_dir, 'mock_gene_expression.csv')
    df_expr.to_csv(expr_path, index=False)
    print(f"Saved mock expression data to {expr_path}")

    # 3. Mock Drug Response Data
    # Columns: COSMIC_ID, CELL_LINE_NAME, DRUG_NAME, LN_IC50
    print(f"Generating mock drug response data...")
    
    response_data = []
    drugs = ['Bortezomib', 'Cisplatin', 'Doxorubicin', 'Gemcitabine', 'Paclitaxel']
    
    for drug in drugs:
        for cid in resp_ids:
            # Random IC50 with some correlation to specific genes (optional, but good for sanity check later)
            # Just random for now
            ic50 = np.random.normal(loc=1, scale=2) 
            
            # Introduce missing values occasionally
            if np.random.random() < 0.05:
                ic50 = np.nan
                
            response_data.append({
                'COSMIC_ID': cid,
                'CELL_LINE_NAME': f"CELL_{cid}",
                'DRUG_NAME': drug,
                'LN_IC50': ic50
            })
            
    df_resp = pd.DataFrame(response_data)
    
    # Save Response Data
    resp_path = os.path.join(output_dir, 'mock_drug_response.csv')
    df_resp.to_csv(resp_path, index=False)
    print(f"Saved mock drug response data to {resp_path}")

if __name__ == "__main__":
    create_mock_gdsc_data()
