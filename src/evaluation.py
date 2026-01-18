import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_prediction_performance(y_true, y_pred, model_name, save_path=None):
    """
    Generates research-quality Predicted vs Actual and Residual plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Predicted vs Actual
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor='k', ax=axes[0])
    
    # Ideal line (Identity)
    min_val = min(y_true.min(), y_pred.min()) - 0.5
    max_val = max(y_true.max(), y_pred.max()) + 0.5
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit')
    
    axes[0].set_xlabel('Actual ln(IC50)', fontsize=12)
    axes[0].set_ylabel('Predicted ln(IC50)', fontsize=12)
    axes[0].set_title(f'{model_name}: Predicted vs Actual', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, edgecolor='k', ax=axes[1])
    axes[1].axhline(0, color='r', linestyle='--', lw=2)
    
    axes[1].set_xlabel('Predicted ln(IC50)', fontsize=12)
    axes[1].set_ylabel('Residuals (Actual - Pred)', fontsize=12)
    axes[1].set_title(f'{model_name}: Residual Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_cv_distribution(cv_scores, model_name):
    """Plots the distribution of Cross-Validation scores."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=cv_scores)
    plt.ylabel('RMSE (Negative)')
    plt.title(f'{model_name} Cross-Validation Score Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()
