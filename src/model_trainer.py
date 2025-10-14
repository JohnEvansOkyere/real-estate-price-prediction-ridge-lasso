"""
Model training and hyperparameter tuning module.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from src import config


class ModelTrainer:
    """Trains and tunes house price prediction models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.preprocessor = self._create_preprocessor()
        self.best_model = None
        self.feature_engineer = None
        
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline.
        
        Returns:
            ColumnTransformer for feature preprocessing
        """
        # Numeric: impute + scale
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Binary: impute only
        binary_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Categorical: impute + one-hot encode
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, config.NUMERIC_FEATURES),
            ('bin', binary_transformer, config.BINARY_FEATURES),
            ('cat', categorical_transformer, config.CATEGORICAL_FEATURES)
        ], remainder='drop')
        
        return preprocessor
    
    def prepare_data(self, df: pd.DataFrame):
        """
        Split data into train/validation sets.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        X = df.drop(columns=[config.TARGET_COL])
        y = df[config.TARGET_COL]
        
        # Apply log transformation if configured
        if config.USE_LOG_TRANSFORM:
            y = np.log1p(y)
            print("✓ Applied log transformation to target")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        print(f"✓ Train set: {X_train.shape}, Validation set: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    
    def train_model(self, X_train, y_train, model_type: str = 'ridge'):
        """
        Train model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: 'ridge' or 'lasso'
            
        Returns:
            Best fitted model
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        # Select model and parameters
        if model_type.lower() == 'ridge':
            model = Ridge(random_state=config.RANDOM_STATE)
            param_grid = {'regressor__alpha': config.RIDGE_ALPHAS}
        elif model_type.lower() == 'lasso':
            model = Lasso(random_state=config.RANDOM_STATE, max_iter=10000)
            param_grid = {'regressor__alpha': config.LASSO_ALPHAS}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=config.CV_FOLDS,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit
        print("Searching for best hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best alpha: {grid_search.best_params_['regressor__alpha']:.6f}")
        print(f"✓ Best CV RMSE: {-grid_search.best_score_:.4f}\n")
        
        return grid_search
    
    def evaluate_model(self, model, X_val, y_val) -> dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target (log-scale if transformed)
            
        Returns:
            Dictionary of metrics
        """
        # Predict
        y_pred_log = model.predict(X_val)
        
        # Transform back if using log
        if config.USE_LOG_TRANSFORM:
            y_val_original = np.expm1(y_val)
            y_pred_original = np.expm1(y_pred_log)
        else:
            y_val_original = y_val
            y_pred_original = y_pred_log
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
        mae = mean_absolute_error(y_val_original, y_pred_original)
        r2 = r2_score(y_val_original, y_pred_original)
        mape = np.mean(np.abs((y_val_original - y_pred_original) / y_val_original)) * 100
        neg_count = (y_pred_original < 0).sum()
        neg_pct = 100 * neg_count / len(y_pred_original)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Negative_Count': neg_count,
            'Negative_Pct': neg_pct
        }
        
        return metrics
    
    def train_and_evaluate(self, df: pd.DataFrame, model_type: str = 'ridge'):
        """
        Complete training and evaluation pipeline.
        
        Args:
            df: DataFrame with features and target
            model_type: 'ridge' or 'lasso'
            
        Returns:
            Trained model and metrics
        """
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(df)
        
        # Train model
        model = self.train_model(X_train, y_train, model_type)
        
        # Evaluate
        print("Evaluating on validation set...")
        metrics = self.evaluate_model(model, X_val, y_val)
        
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"  RMSE:     {metrics['RMSE']:.4f} lakhs")
        print(f"  MAE:      {metrics['MAE']:.4f} lakhs")
        print(f"  R²:       {metrics['R2']:.4f}")
        print(f"  MAPE:     {metrics['MAPE']:.2f}%")
        print(f"  Negative: {metrics['Negative_Count']} ({metrics['Negative_Pct']:.2f}%)")
        
        if metrics['Negative_Pct'] == 0:
            print("  ✓ No negative predictions!")
        
        print(f"{'='*60}\n")
        
        self.best_model = model.best_estimator_
        return model, metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump(self.best_model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model.
        
        Args:
            filepath: Path to model file
        """
        self.best_model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")


def main():
    """Example usage of ModelTrainer."""
    from src.data_preprocessor import DataProcessor
    from src.feature_engineer import FeatureEngineer
    
    # Load and process data
    processor = DataProcessor()
    train_df = processor.process(config.TRAIN_PATH, is_train=True)
    
    # Feature engineering
    engineer = FeatureEngineer()
    train_features = engineer.fit_transform(train_df)
    
    # Train model
    trainer = ModelTrainer()
    model, metrics = trainer.train_and_evaluate(train_features, model_type='ridge')
    
    # Save model
    trainer.save_model(config.MODEL_PATH)
    
    return model, metrics


if __name__ == "__main__":
    main()