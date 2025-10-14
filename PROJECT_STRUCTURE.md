# House Price Prediction - ML Project Structure

```
house-price-prediction/
│
├── data/
│   ├── raw/                    # Original data files
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/              # Cleaned data (generated)
│       └── train_cleaned.csv
│
├── models/                     # Saved models (generated)
│   └── ridge_model.pkl
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py              # All configurations
│   ├── data_processor.py      # Data cleaning & validation
│   ├── feature_engineer.py    # Feature engineering
│   ├── model_trainer.py       # Model training & tuning
│   ├── predictor.py           # Prediction pipeline
│   └── utils.py               # Helper functions
│
├── notebooks/                  # Jupyter notebooks (your work)
│   └── exploration.ipynb
│
├── main.py                     # Main pipeline script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
python main.py --mode train

# 3. Make predictions
python main.py --mode predict

# 4. Full pipeline (train + predict)
python main.py --mode pipeline
```