# Data Dictionary & Schema

## 1. Drug Response Data (Target)

**Source:** GDSC (Genomics of Drug Sensitivity in Cancer)
**File:** `PANCANCER_IC_Fri_Aug_14_13_16_56_2020.csv` (Example filename from GDSC2)

| Column Name | Data Type | Description |

|-------------|-----------|-------------|
| `COSMIC_ID` | Integer | Unique identifier for the cell line (Key) |
| `CELL_LINE_NAME`| String | Common name of the cancer cell line |
| `DRUG_NAME` | String | Name of the compound |
| `LN_IC50` | Float | Natural log of the half-maximal inhibitory concentration. Lower values = Higher sensitivity. |

## 2. Gene Expression Data (Features)

**Source:** GDSC / RMA Normalized Affymetrix
**File:** `Cell_line_RMA_proc_basalExp.txt`

| Column Name | Data Type | Description |
|-------------|-----------|-------------|

| `GENE_SYMBOLS` | String | HUGO Gene Nomenclature Committee (HGNC) gene symbol |
| `COSMIC_ID_{ID}`| Float | Expression value for cell line with `COSMIC_ID` (Columns are samples) |
| `GENE_ID` | String | Ensembl ID or other identifier |

## 3. Schema Relationship

erDiagram
    CELL_LINE ||--|| DRUG_RESPONSE : has
    CELL_LINE ||--|| GENE_EXPRESSION : possesses

    CELL_LINE {
        int COSMIC_ID PK
        string Name
        string Tissue_Type
    }
    
    DRUG_RESPONSE {
        int COSMIC_ID FK
        string Drug_Name
        float LN_IC50
    }
    
    GENE_EXPRESSION {
        int COSMIC_ID FK
        float Gene_1_Exp
        float Gene_N_Exp
    }

## Biological Constraints

- **Matching:** We must take the intersection of `COSMIC_ID` present in both the Expression matrix and the Drug Response table.
- **Drug Selection:** For this project, we will filter for a **single drug** (e.g., *Bortezomib* or *Doxorubicin*) to create a scalar regression target vector.
