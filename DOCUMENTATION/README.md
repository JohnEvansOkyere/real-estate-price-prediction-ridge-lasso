# Machine Learning Price Prediction Project

A comprehensive, professional machine learning project implementing **OLS**, **Ridge**, and **Lasso** regression for price prediction with detailed documentation and explanations.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Model Explanations](#model-explanations)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎯 Project Overview

This project implements three fundamental regression algorithms to predict prices:

1. **Ordinary Least Squares (OLS)** - Baseline linear regression
2. **Ridge Regression** - L2 regularization for handling multicollinearity
3. **Lasso Regression** - L1 regularization with automatic feature selection

### Key Features

✅ **Complete ML Pipeline**: From data preprocessing to model deployment  
✅ **Automated Hyperparameter Tuning**: Cross-validation for optimal parameters  
✅ **Comprehensive Documentation**: Detailed explanations of all decisions  
✅ **Professional Code**: Industry-standard practices with extensive comments  
✅ **Model Comparison**: Automated evaluation and selection of best model  
✅ **Reproducible Results**: Fixed random seeds for consistency  

---

## 📁 Project Structure

```
price-prediction/
│
├── config.py                  # Centralized configuration
├── preprocessing.py           # Data preparation module
├── models.py                  # Regression models implementation
├── train.py                   # Main training pipeline
├── predict.py                 # Prediction script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── datasets/
│   ├── raw/
│   │   ├── train.csv         # Training data
│   │   └── test.csv          # Test data
│   └── processed/            # Preprocessed data (generated)
│
├── models/                    # Saved models (generated)
│   ├── ols_model.pkl
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── results/                   # Results and reports (generated)
│   ├── model_comparison.csv
│   ├── test_predictions.csv
│   ├── final_report.txt
│   └── training.log
│
└── notebooks/                 # Jupyter notebooks
    └── exploratory_analysis.ipynb
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd price-prediction

# Or simply download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Prepare Your Data

Place your CSV files in the `datasets/raw/` directory:
- `train.csv` - Training dataset with target column
- `test.csv` - Test dataset (with or without target column)

**Required format:**
- Must include a column named `TARGET(Price)` (configurable in `config.py`)
- Can include any number of numerical and categorical features

### 2. Train Models

```bash
python train.py
```

This will:
- Load and preprocess the data
- Train OLS, Ridge, and Lasso models
- Tune hyperparameters automatically
- Compare model performance
- Select and save the best model
- Generate a comprehensive report

**Expected output:**
```
================================================================================
STARTING MACHINE LEARNING TRAINING PIPELINE
================================================================================
...
Best model: Ridge Regression
Validation R²: 0.8543
All results saved to: results/
```

### 3. Make Predictions

```bash
python predict.py
```

This will:
- Load the best model
- Generate predictions on test data
- Save predictions to `results/test_predictions.csv`
- Create a prediction summary report

---

## 📖 Detailed Usage

### Configuration

All project settings are in `config.py`. Key configurations:

```python
# Data settings
TARGET_COLUMN = "TARGET(Price)"
MISSING_VALUE_STRATEGY = {"numerical": "median", "categorical": "mode"}
SCALING_METHOD = "standard"  # Options: "standard", "minmax", "robust"

# Model settings
CV_FOLDS = 5
RANDOM_STATE = 42

# Hyperparameter ranges
RIDGE_ALPHAS = np.logspace(-3, 3, 50)
LASSO_ALPHAS = np.logspace(-3, 3, 50)
```

### Data Preprocessing

The preprocessing pipeline (`preprocessing.py`) handles:

1. **Missing Values**
   - Numerical: Median imputation (configurable)
   - Categorical: Most frequent value

2. **Categorical Encoding**
   - OneHot encoding (drops first category to avoid multicollinearity)
   - Handles unknown categories in test data

3. **Feature Scaling**
   - Standard scaling (mean=0, std=1) by default
   - Essential for regularized models (Ridge/Lasso)

**Justification:** 
- Missing value imputation preserves sample size
- Scaling ensures fair coefficient comparison in regularized models
- OneHot encoding enables linear models to handle categorical data

### Using Individual Modules

```python
# Load and preprocess data
from preprocessing import DataPreprocessor, load_data

train_df, test_df = load_data()
preprocessor = DataPreprocessor()
X_train, y_train = preprocessor.fit_transform(train_df)
X_test = preprocessor.transform(test_df)

# Train a specific model
from models import RidgeRegression

ridge = RidgeRegression()
ridge.tune_hyperparameters(X_train, y_train)
ridge.train(X_train, y_train)
predictions = ridge.predict(X_test)

# Evaluate
metrics = ridge.evaluate(X_test, y_test)
print(f"R²: {metrics['r2']:.4f}")
```

### Jupyter Notebook Analysis

For exploratory data analysis, use the provided notebook:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 🧠 Model Explanations

### 1. Ordinary Least Squares (OLS)

**Mathematical Principle:**
```
Minimize: Σ(y - ŷ)²
Solution: β = (X'X)⁻¹X'y
```

**Strengths:**
- Simple and interpretable
- Fast computation
- No hyperparameters

**Weaknesses:**
- Sensitive to multicollinearity
- Prone to overfitting with many features
- No feature selection

**Best Used When:**
- You have few, uncorrelated features
- Maximum interpretability is needed
- As a baseline for comparison

---

### 2. Ridge Regression (L2 Regularization)

**Mathematical Principle:**
```
Minimize: Σ(y - ŷ)² + α·Σ(βⱼ²)
```

**Strengths:**
- Handles multicollinearity excellently
- More stable than OLS
- Keeps all features
- Smooth coefficient shrinkage

**Weaknesses:**
- Does not perform feature selection
- Includes all features (even irrelevant ones)
- Requires tuning α

**Best Used When:**
- Features are correlated
- All features are potentially relevant
- You want a stable, robust model
- Prediction accuracy > interpretability

**Hyperparameter α:**
- Small α → closer to OLS (less regularization)
- Large α → more shrinkage (simpler model)
- Tuned via cross-validation in this project

---

### 3. Lasso Regression (L1 Regularization)

**Mathematical Principle:**
```
Minimize: Σ(y - ŷ)² + α·Σ|βⱼ|
```

**Strengths:**
- Automatic feature selection (sets coefficients to zero)
- Produces sparse, interpretable models
- Excellent for high-dimensional data
- Identifies key features

**Weaknesses:**
- Arbitrary selection among correlated features
- Can be unstable with high correlation
- May exclude useful correlated features

**Best Used When:**
- Many features are irrelevant
- You need feature selection
- Interpretability with few features is desired
- You want to identify key drivers

**Hyperparameter α:**
- Small α → more features retained
- Large α → fewer features (more sparsity)
- Tuned via cross-validation in this project

---

## 📊 Results

After training, check these files:

### 1. Model Comparison (`results/model_comparison.csv`)

Example output:
```
model               r2      adjusted_r2  rmse      mae       mape
OLS Regression      0.8234  0.8198      12.45     9.23      8.56
Ridge Regression    0.8543  0.8512      11.32     8.45      7.82
Lasso Regression    0.8489  0.8461      11.54     8.67      7.95
```

### 2. Final Report (`results/final_report.txt`)

Comprehensive report including:
- Data preprocessing decisions with justifications
- Detailed model explanations
- Performance comparison
- Model selection rationale
- Top important features
- Production recommendations

### 3. Predictions (`results/test_predictions.csv`)

Contains:
- Original test data
- Predicted prices
- Residuals (if test labels available)
- Absolute and percentage errors

---

## 📈 Performance Metrics Explained

- **R² (R-squared)**: Proportion of variance explained [0 to 1, higher is better]
  - 0.0 = model explains nothing
  - 1.0 = perfect predictions
  - 0.8+ = very good model

- **Adjusted R²**: R² penalized for number of features
  - Better for comparing models with different feature counts
  - Prevents overfitting rewards

- **RMSE (Root Mean Squared Error)**: Average prediction error in target units
  - Lower is better
  - Sensitive to large errors
  - Same units as target variable

- **MAE (Mean Absolute Error)**: Average absolute error
  - Lower is better
  - Less sensitive to outliers than RMSE
  - More interpretable

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
  - Lower is better
  - Scale-independent
  - Good for comparison across datasets

---

## 🎓 Understanding Model Selection

The training pipeline automatically selects the best model based on R². The final report explains why:

**If OLS wins:**
- Features have low multicollinearity
- Sample size is adequate
- No overfitting detected

**If Ridge wins:**
- Features are correlated
- Regularization improves generalization
- All features contribute

**If Lasso wins:**
- Many features are irrelevant
- Sparse solution performs better
- Feature selection was beneficial

---

## 🔍 Troubleshooting

### Common Issues

**1. File Not Found Error**
```
FileNotFoundError: datasets/raw/train.csv
```
**Solution:** Ensure `train.csv` and `test.csv` are in `datasets/raw/`

**2. Target Column Not Found**
```
KeyError: 'TARGET(Price)'
```
**Solution:** Update `TARGET_COLUMN` in `config.py` to match your data

**3. Memory Error with Large Datasets**
```
MemoryError: Unable to allocate array
```
**Solution:** 
- Reduce `RIDGE_ALPHAS` and `LASSO_ALPHAS` array sizes in `config.py`
- Use a subset of data for initial testing

**4. Poor Model Performance**
```
R² is negative or very low
```
**Solution:**
- Check data quality (missing values, outliers)
- Verify target variable is continuous
- Consider feature engineering
- Check for data leakage

---

## 🧪 Testing

To verify the installation:

```python
python -c "import pandas, sklearn, numpy; print('All dependencies installed successfully!')"
```

To test with sample data, create a simple CSV:

```python
import pandas as pd
import numpy as np

# Create sample training data
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
    'TARGET(Price)': np.random.randn(n_samples) * 10 + 50
})
df.to_csv('datasets/raw/train.csv', index=False)

# Create sample test data
df_test = df.copy().drop(columns=['TARGET(Price)'])
df_test.to_csv('datasets/raw/test.csv', index=False)

print("Sample data created!")
```

Then run: `python train.py`

---

## 🤝 Contributing

Improvements welcome! Areas for contribution:

- Additional regression algorithms (Elastic Net, XGBoost, etc.)
- Enhanced visualizations
- Automated feature engineering
- Model deployment scripts
- Unit tests
- Documentation improvements

---

## 📝 License

This project is for educational purposes.

---

## 👨‍💻 Author

**ML Engineering Team**  
Professional Machine Learning Implementation  
Date: 2025-10-14

---

## 🙏 Acknowledgments

Built with:
- scikit-learn: Machine learning algorithms
- pandas: Data manipulation
- NumPy: Numerical computing
- matplotlib/seaborn: Visualization

---

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Ridge vs Lasso: Understanding the Difference](https://scikit-learn.org/stable/modules/linear_model.html)
- [Cross-Validation Explained](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Feature Engineering Guide](https://scikit-learn.org/stable/modules/preprocessing.html)

---

**Happy Modeling! 🚀**