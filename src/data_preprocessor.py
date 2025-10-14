"""
Data cleaning and validation module.
Handles outlier removal and data quality checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src import config


class DataProcessor:
    """Cleans and validates raw housing data."""
    
    def __init__(self):
        """Initialize data processor with config settings."""
        self.square_ft_min = config.SQUARE_FT_MIN
        self.square_ft_max = config.SQUARE_FT_MAX
        self.bhk_min = config.BHK_MIN
        self.bhk_max = config.BHK_MAX
        self.price_iqr_mult = config.PRICE_IQR_MULTIPLIER
        
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """
        Load CSV data.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"✓ Loaded data: {df.shape}")
        return df
    
    def remove_outliers(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Remove outliers from training data or cap them in test data.
        
        Args:
            df: Input DataFrame
            is_train: If True, remove outliers; if False, cap them
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        initial_shape = df.shape[0]
        
        if is_train:
            # Remove outliers from training data
            df = df[
                (df['SQUARE_FT'] >= self.square_ft_min) &
                (df['SQUARE_FT'] <= self.square_ft_max) &
                (df['BHK_NO.'] >= self.bhk_min) &
                (df['BHK_NO.'] <= self.bhk_max)
            ].copy()
            
            # Remove price outliers (only for training data)
            if config.TARGET_COL in df.columns:
                Q1 = df[config.TARGET_COL].quantile(0.25)
                Q3 = df[config.TARGET_COL].quantile(0.75)
                IQR = Q3 - Q1
                upper_limit = Q3 + self.price_iqr_mult * IQR
                df = df[df[config.TARGET_COL] <= upper_limit].copy()
            
            removed = initial_shape - df.shape[0]
            print(f"✓ Removed {removed} outliers ({100*removed/initial_shape:.1f}%)")
            
        else:
            # Cap outliers in test data (don't remove, we need predictions)
            df['SQUARE_FT'] = df['SQUARE_FT'].clip(self.square_ft_min, self.square_ft_max)
            df['BHK_NO.'] = df['BHK_NO.'].clip(self.bhk_min, self.bhk_max)
            print(f"✓ Capped extreme values in test data")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If critical issues found
        """
        # Check for required columns
        required_cols = ['SQUARE_FT', 'BHK_NO.', 'POSTED_BY']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for completely null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            raise ValueError(f"Columns with all null values: {null_cols}")
        
        print(f"✓ Data validation passed")
    
    def process(self, filepath: Path, is_train: bool = True) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Args:
            filepath: Path to data file
            is_train: Whether this is training data
            
        Returns:
            Processed DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Processing {'training' if is_train else 'test'} data")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_data(filepath)
        
        # Validate
        self.validate_data(df)
        
        # Remove/cap outliers
        df = self.remove_outliers(df, is_train=is_train)
        
        print(f"✓ Final shape: {df.shape}\n")
        return df


def main():
    """Example usage of DataProcessor."""
    processor = DataProcessor()
    
    # Process training data
    train_df = processor.process(config.TRAIN_PATH, is_train=True)
    train_df.to_csv(config.TRAIN_CLEANED_PATH, index=False)
    print(f"✓ Saved cleaned data to {config.TRAIN_CLEANED_PATH}")
    
    # Process test data
    test_df = processor.process(config.TEST_PATH, is_train=False)
    
    return train_df, test_df


if __name__ == "__main__":
    main()