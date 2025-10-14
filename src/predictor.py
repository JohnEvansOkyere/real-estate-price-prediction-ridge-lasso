"""
Prediction module for making house price predictions.
"""

import pandas as pd
import numpy as np
import joblib
from src import config


class Predictor:
    """Makes predictions on new data using trained model."""
    
    def __init__(self, model_path: str = None, feature_engineer_path: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model (optional)
            feature_engineer_path: Path to feature engineer (optional)
        """
        self.model = None
        self.feature_engineer = None
        
        if model_path:
            self.load_model(model_path)
        if feature_engineer_path:
            self.load_feature_engineer(feature_engineer_path)
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model.
        
        Args:
            filepath: Path to model file
        """
        self.model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
    
    def load_feature_engineer(self, filepath: str) -> None:
        """
        Load feature engineer (KMeans).
        
        Args:
            filepath: Path to feature engineer file
        """
        from src.feature_engineer import FeatureEngineer
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.load(filepath)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: DataFrame with features
            
        Returns:
            Array of predictions (in lakhs)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")
        
        # Predict (in log scale if model was trained on log)
        predictions_log = self.model.predict(X)
        
        # Transform back if using log
        if config.USE_LOG_TRANSFORM:
            predictions = np.expm1(predictions_log)
        else:
            predictions = predictions_log
        
        return predictions
    
    def predict_single(self, property_dict: dict) -> float:
        """
        Predict price for a single property.
        
        Args:
            property_dict: Dictionary with property features
            
        Returns:
            Predicted price in lakhs
        """
        # Convert to DataFrame
        df = pd.DataFrame([property_dict])
        
        # Predict
        prediction = self.predict(df)[0]
        
        return prediction
    
    def predict_from_file(self, filepath: str, output_path: str = None) -> pd.DataFrame:
        """
        Make predictions from CSV file.
        
        Args:
            filepath: Path to input CSV
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        print(f"\n{'='*60}")
        print("Making Predictions")
        print(f"{'='*60}")
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} samples")
        
        # Make predictions
        predictions = self.predict(df)
        
        # Check for issues
        neg_count = (predictions < 0).sum()
        if neg_count > 0:
            print(f"⚠️  {neg_count} negative predictions found!")
        else:
            print("✓ All predictions are positive")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Predicted_Price': predictions
        })
        
        # Save if output path provided
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"✓ Predictions saved to {output_path}")
        
        print(f"\nPrediction Statistics:")
        print(results['Predicted_Price'].describe())
        print(f"{'='*60}\n")
        
        return results
    
    def validate_prediction(self, square_ft: float, predicted_price: float) -> str:
        """
        Validate if prediction seems reasonable.
        
        Args:
            square_ft: Property square footage
            predicted_price: Predicted price
            
        Returns:
            Validation message
        """
        price_per_1000sqft = (predicted_price / square_ft) * 1000
        
        if price_per_1000sqft < 10:
            return "⚠️  Price seems too low for size (possible data error)"
        elif price_per_1000sqft > 200:
            return "⚠️  Price seems very high (ultra-luxury or error)"
        else:
            return "✓ Prediction seems reasonable"


def main():
    """Example usage of Predictor."""
    from src.data_preprocessor import DataProcessor
    from src.feature_engineer import FeatureEngineer
    
    # Load and process test data
    processor = DataProcessor()
    test_df = processor.process(config.TEST_PATH, is_train=False)
    
    # Load feature engineer and transform
    engineer = FeatureEngineer()
    engineer.load(config.MODEL_DIR / "feature_engineer.pkl")
    test_features = engineer.transform(test_df)
    
    # Initialize predictor
    predictor = Predictor(model_path=config.MODEL_PATH)
    
    # Make predictions
    predictions = predictor.predict(test_features)
    
    # Save results
    results = pd.DataFrame({
        'Id': range(len(predictions)),
        'Predicted_Price': predictions
    })
    results.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"✓ Predictions saved to {config.PREDICTIONS_PATH}")
    
    # Example: Predict single property
    sample_property = {
        'SQUARE_FT': 1200,
        'BHK_NO.': 2,
        'POSTED_BY': 'Owner',
        'UNDER_CONSTRUCTION': 0,
        'RERA': 1,
        'READY_TO_MOVE': 1,
        'RESALE': 0,
        'LONGITUDE': 77.5946,
        'LATITUDE': 12.9716,
        'ADDRESS': 'Bangalore, Karnataka'
    }
    
    # Process single prediction (would need feature engineering)
    print("\nExample single prediction:")
    print(f"Input: {sample_property}")
    # price = predictor.predict_single(sample_property)
    # print(f"Predicted Price: {price:.2f} lakhs")
    
    return predictions


if __name__ == "__main__":
    main()