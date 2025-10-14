"""
Feature engineering module.
Creates and transforms features for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
from src import config


class FeatureEngineer:
    """Creates features from raw data."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.kmeans = None
        
    def parse_city(self, address: str) -> str:
        """
        Extract city name from address string.
        
        Args:
            address: Full address string
            
        Returns:
            City name (lowercase)
        """
        if pd.isna(address):
            return "unknown"
        
        parts = [p.strip() for p in str(address).split(',') if p.strip()]
        if len(parts) == 0:
            return "unknown"
        
        return parts[-1].lower()
    
    def create_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create numeric features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new numeric features
        """
        df = df.copy()
        
        # Feature 1: Square feet per bedroom
        df['sqft_per_bhk'] = df['SQUARE_FT'] / df['BHK_NO.'].replace(0, np.nan)
        
        # Feature 2: Log-transformed square feet
        df['log_sqft'] = np.log1p(df['SQUARE_FT'])
        
        # Cap sqft_per_bhk to reasonable range
        df['sqft_per_bhk'] = df['sqft_per_bhk'].clip(200, 2000)
        
        return df
    
    def create_geographic_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create geographic clustering features.
        
        Args:
            df: Input DataFrame
            fit: If True, fit new KMeans; if False, use existing
            
        Returns:
            DataFrame with geo_cluster feature
        """
        df = df.copy()
        
        # Extract coordinates
        coords = df[['LONGITUDE', 'LATITUDE']].copy()
        coords = coords.fillna(coords.median())
        
        if fit:
            # Fit new KMeans clustering
            self.kmeans = KMeans(
                n_clusters=config.N_GEO_CLUSTERS,
                random_state=config.RANDOM_STATE,
                n_init=10
            )
            df['geo_cluster'] = self.kmeans.fit_predict(coords)
        else:
            # Use existing KMeans
            if self.kmeans is None:
                raise ValueError("KMeans not fitted. Call with fit=True first.")
            df['geo_cluster'] = self.kmeans.predict(coords)
        
        # Convert to string for categorical encoding
        df['geo_cluster'] = 'cluster_' + df['geo_cluster'].astype(str)
        
        return df
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed categorical features
        """
        df = df.copy()
        
        # Merge Builder into Dealer
        df['POSTED_BY'] = df['POSTED_BY'].replace(config.POSTED_BY_MAPPING)
        
        # Parse city from address
        if 'ADDRESS' in df.columns:
            df['city'] = df['ADDRESS'].apply(self.parse_city)
        
        # Drop features
        df = df.drop(columns=config.DROP_FEATURES, errors='ignore')
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit feature engineering pipeline and transform data.
        Use this for TRAINING data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Transformed DataFrame
        """
        print(f"\n{'='*60}")
        print("Feature Engineering (Fit & Transform)")
        print(f"{'='*60}")
        
        df = self.create_categorical_features(df)
        print("✓ Created categorical features")
        
        df = self.create_numeric_features(df)
        print("✓ Created numeric features")
        
        df = self.create_geographic_features(df, fit=True)
        print(f"✓ Created {config.N_GEO_CLUSTERS} geographic clusters")
        
        print(f"✓ Final features: {df.shape[1]} columns\n")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline.
        Use this for TEST data.
        
        Args:
            df: Test DataFrame
            
        Returns:
            Transformed DataFrame
        """
        print(f"\n{'='*60}")
        print("Feature Engineering (Transform Only)")
        print(f"{'='*60}")
        
        df = self.create_categorical_features(df)
        print("✓ Created categorical features")
        
        df = self.create_numeric_features(df)
        print("✓ Created numeric features")
        
        df = self.create_geographic_features(df, fit=False)
        print("✓ Applied geographic clusters")
        
        print(f"✓ Final features: {df.shape[1]} columns\n")
        return df
    
    def save(self, filepath: str) -> None:
        """
        Save fitted feature engineer (KMeans model).
        
        Args:
            filepath: Path to save file
        """
        if self.kmeans is None:
            raise ValueError("No fitted KMeans to save. Call fit_transform first.")
        
        joblib.dump(self.kmeans, filepath)
        print(f"✓ Saved feature engineer to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load fitted feature engineer.
        
        Args:
            filepath: Path to load file
        """
        self.kmeans = joblib.load(filepath)
        print(f"✓ Loaded feature engineer from {filepath}")


def main():
    """Example usage of FeatureEngineer."""
    from src.data_preprocessor import DataProcessor
    
    # Load and clean data
    processor = DataProcessor()
    train_df = processor.process(config.TRAIN_PATH, is_train=True)
    
    # Feature engineering
    engineer = FeatureEngineer()
    train_features = engineer.fit_transform(train_df)
    
    print("Sample features:")
    print(train_features[['SQUARE_FT', 'sqft_per_bhk', 'log_sqft', 
                          'city', 'geo_cluster']].head())
    
    return train_features


if __name__ == "__main__":
    main()