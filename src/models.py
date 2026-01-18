import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    """
    Manages training and evaluation of regression models for Drug Response Prediction.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_baseline(self):
        """Trains a simple Linear Regression model."""
        print("Training Baseline (Linear Regression)...")
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        self.models['LinearRegression'] = model
        return model

    def train_random_forest(self, n_estimators=100, max_depth=None):
        """Trains a Random Forest Regressor."""
        print("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = model
        return model
        
    def train_xgboost(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """Trains an XGBoost Regressor."""
        print("Training XGBoost...")
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            missing=np.nan # Handle missing values natively if any exist
        )
        model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = model
        return model
    
    def evaluate_model(self, model_name):
        """
        Evaluates a specific model using RMSE, R2, and Cross-Validation.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found! Train it first.")
            
        model = self.models[model_name]
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        # Cross-Validation (5-fold)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores.mean()
        
        metrics = {
            'Train RMSE': train_rmse,
            'Test RMSE': test_rmse,
            'CV RMSE': cv_rmse,
            'Test R2': test_r2
        }
        
        self.results[model_name] = metrics
        return metrics

    def get_comparison_table(self):
        """Returns a DataFrame comparing all trained models."""
        return pd.DataFrame(self.results).T
