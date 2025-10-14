"""
Utility functions for the project.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def print_section(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of header
    """
    print(f"\n{'='*width}")
    print(title.center(width))
    print(f"{'='*width}\n")


def save_metrics(metrics: dict, filepath: Path) -> None:
    """
    Save model metrics to CSV.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Output file path
    """
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"✓ Metrics saved to {filepath}")


def load_metrics(filepath: Path) -> dict:
    """
    Load metrics from CSV.
    
    Args:
        filepath: Path to metrics file
        
    Returns:
        Dictionary of metrics
    """
    df = pd.read_csv(filepath)
    return df.iloc[0].to_dict()


def plot_predictions(y_true, y_pred, title: str = "Predictions vs Actual",
                     save_path: Path = None) -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price (lakhs)', fontsize=12)
    plt.ylabel('Predicted Price (lakhs)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, title: str = "Residual Plot",
                   save_path: Path = None) -> None:
    """
    Plot residuals.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot (optional)
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_pred, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    
    plt.xlabel('Predicted Price (lakhs)', fontsize=12)
    plt.ylabel('Residuals (lakhs)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()


def generate_report(metrics: dict, save_path: Path = None) -> str:
    """
    Generate a text report of model performance.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save report (optional)
        
    Returns:
        Report string
    """
    report = f"""
    {'='*60}
    HOUSE PRICE PREDICTION MODEL - PERFORMANCE REPORT
    {'='*60}

    Model Metrics:
    --------------
    RMSE:                   {metrics['RMSE']:.4f} lakhs
    MAE:                    {metrics['MAE']:.4f} lakhs
    R²:                     {metrics['R2']:.4f}
    MAPE:                   {metrics['MAPE']:.2f}%
    Negative Predictions:   {metrics['Negative_Count']} ({metrics['Negative_Pct']:.2f}%)

    Interpretation:
    ---------------
    - Average error: {metrics['MAE']:.2f} lakhs
    - Explains {metrics['R2']*100:.1f}% of price variance
    - Typical prediction is within {metrics['MAPE']:.1f}% of actual price

    Status: {'✓ PRODUCTION READY' if metrics['Negative_Pct'] == 0 else '⚠️  NEEDS REVIEW'}

    {'='*60}
    """
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to {save_path}")
    
    return report


def check_data_quality(df: pd.DataFrame) -> None:
    """
    Print data quality summary.
    
    Args:
        df: DataFrame to check
    """
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  ✓ No missing values")
    else:
        print(missing)
    
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    print("\n" + "="*60 + "\n")