"""
Main pipeline script for house price prediction.

Usage:
    python main.py --mode train          # Train model
    python main.py --mode predict        # Make predictions
    python main.py --mode pipeline       # Full pipeline (train + predict)
"""

import argparse
from pathlib import Path
from src import config
from src.data_preprocessor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor
from src.utils import print_section, generate_report


def train_pipeline():
    """Complete training pipeline."""
    print_section("HOUSE PRICE PREDICTION - TRAINING PIPELINE")
    
    # Step 1: Process data
    processor = DataProcessor()
    train_df = processor.process(config.TRAIN_PATH, is_train=True)
    
    # Step 2: Feature engineering
    engineer = FeatureEngineer()
    train_features = engineer.fit_transform(train_df)
    
    # Save feature engineer
    feature_engineer_path = config.MODEL_DIR / "feature_engineer.pkl"
    engineer.save(feature_engineer_path)
    
    # Step 3: Train model
    trainer = ModelTrainer()
    
    # Prepare data (this returns X_train, X_val, y_train, y_val)
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    X = train_features.drop(columns=[config.TARGET_COL])
    y = train_features[config.TARGET_COL]
    
    # Apply log transformation if configured
    if config.USE_LOG_TRANSFORM:
        y = np.log1p(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # Train model
    model = trainer.train_model(X_train, y_train, model_type='ridge')
    
    # Step 4: Comprehensive Evaluation
    from src.evaluator import ModelEvaluator
    
    evaluator = ModelEvaluator(
        model=model.best_estimator_,
        X_val=X_val,
        y_val=y_val,
        model_name='Ridge (Log-Transformed)'
    )
    
    metrics = evaluator.generate_full_report()
    
    # Step 5: Save model
    trainer.best_model = model.best_estimator_
    trainer.save_model(config.MODEL_PATH)
    
    print_section("✓ TRAINING COMPLETE")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Feature engineer saved to: {feature_engineer_path}")
    print(f"Evaluation saved to: {config.OUTPUTS_DIR}")
    
    return model, metrics


def predict_pipeline():
    """Complete prediction pipeline."""
    print_section("HOUSE PRICE PREDICTION - PREDICTION PIPELINE")
    
    # Check if model exists
    if not config.MODEL_PATH.exists():
        print("❌ No trained model found!")
        print("Please run: python main.py --mode train")
        return None
    
    # Step 1: Process test data
    processor = DataProcessor()
    test_df = processor.process(config.TEST_PATH, is_train=False)
    
    # Step 2: Feature engineering
    engineer = FeatureEngineer()
    feature_engineer_path = config.MODEL_DIR / "feature_engineer.pkl"
    
    if not feature_engineer_path.exists():
        print("❌ Feature engineer not found!")
        print("Please run: python main.py --mode train")
        return None
    
    engineer.load(feature_engineer_path)
    test_features = engineer.transform(test_df)
    
    # Step 3: Load model and predict
    predictor = Predictor(model_path=config.MODEL_PATH)
    predictions = predictor.predict(test_features)
    
    # Step 4: Save predictions
    import pandas as pd
    results = pd.DataFrame({
        'Id': range(len(predictions)),
        'Predicted_Price': predictions
    })
    results.to_csv(config.PREDICTIONS_PATH, index=False)
    
    print_section("✓ PREDICTIONS COMPLETE")
    print(f"Predictions saved to: {config.PREDICTIONS_PATH}")
    print(f"\nPrediction Statistics:")
    print(results['Predicted_Price'].describe())
    
    # Check for negative predictions
    neg_count = (predictions < 0).sum()
    if neg_count > 0:
        print(f"\n⚠️  {neg_count} negative predictions found!")
    else:
        print("\n✓ All predictions are positive!")
    
    return results


def full_pipeline():
    """Run complete pipeline: train + predict."""
    print_section("HOUSE PRICE PREDICTION - FULL PIPELINE")
    
    # Train
    model, metrics = train_pipeline()
    
    # Predict
    results = predict_pipeline()
    
    print_section("✓ FULL PIPELINE COMPLETE")
    
    return model, metrics, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='House Price Prediction Pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'pipeline'],
        default='pipeline',
        help='Pipeline mode: train, predict, or full pipeline'
    )
    
    args = parser.parse_args()
    
    # Run selected pipeline
    if args.mode == 'train':
        train_pipeline()
    elif args.mode == 'predict':
        predict_pipeline()
    elif args.mode == 'pipeline':
        full_pipeline()


if __name__ == "__main__":
    main()