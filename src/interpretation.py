import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelInterpreter:
    """
    Extracts and visualizes biological feature importance from trained models.
    """
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def get_feature_importance(self):
        """
        Extracts feature importance. Supports XGBoost, RandomForest, and Linear Regression (coefficients).
        """
        importance = None
        
        # XGBoost / Random Forest
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        # Linear Regression
        elif hasattr(self.model, 'coef_'):
            importance = abs(self.model.coef_)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute.")
            
        df_imp = pd.DataFrame({
            'Gene': self.feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        return df_imp
    
    def plot_top_genes(self, top_n=20, save_path=None):
        """
        Plots the top N most important genes driving prediction.
        """
        df_imp = self.get_feature_importance().head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Gene', data=df_imp, palette='viridis')
        plt.title(f'Top {top_n} Genes Determining Drug Sensitivity')
        plt.xlabel('Relative Feature Importance')
        plt.ylabel('Gene Symbol')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
        return df_imp
