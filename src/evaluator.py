"""
Comprehensive model evaluation module.
Generates visualizations, metrics, and reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from src import config


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations and reports."""
    
    def __init__(self, model, X_val, y_val, model_name: str = "Ridge"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target (in log scale if transformed)
            model_name: Name of the model for reports
        """
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.model_name = model_name
        self.predictions = None
        self.metrics = None
        
        # Make predictions
        self._make_predictions()
        
    def _make_predictions(self):
        """Make predictions and transform back from log scale if needed."""
        y_pred_log = self.model.predict(self.X_val)
        
        # Transform back if using log
        if config.USE_LOG_TRANSFORM:
            self.y_val_original = np.expm1(self.y_val)
            self.y_pred_original = np.expm1(y_pred_log)
        else:
            self.y_val_original = self.y_val
            self.y_pred_original = y_pred_log
        
        # Convert to Series for easier indexing
        self.predictions = pd.Series(self.y_pred_original, index=self.y_val.index)
    
    def calculate_metrics(self) -> dict:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        rmse = np.sqrt(mean_squared_error(self.y_val_original, self.y_pred_original))
        mae = mean_absolute_error(self.y_val_original, self.y_pred_original)
        r2 = r2_score(self.y_val_original, self.y_pred_original)
        mape = np.mean(np.abs((self.y_val_original - self.y_pred_original) / self.y_val_original)) * 100
        
        neg_count = (self.y_pred_original < 0).sum()
        neg_pct = 100 * neg_count / len(self.y_pred_original)
        
        self.metrics = {
            'Model': self.model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Negative_Count': int(neg_count),
            'Negative_Pct': neg_pct
        }
        
        return self.metrics
    
    def plot_predictions_vs_actual(self, save_path: Path = None):
        """
        Plot predictions vs actual values.
        
        Args:
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(self.y_val_original, self.y_pred_original, 
                   alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(self.y_val_original.min(), self.y_pred_original.min())
        max_val = max(self.y_val_original.max(), self.y_pred_original.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price (lakhs)', fontsize=13, fontweight='bold')
        plt.ylabel('Predicted Price (lakhs)', fontsize=13, fontweight='bold')
        plt.title(f'{self.model_name}: Predictions vs Actual\nR² = {self.metrics["R2"]:.4f}', 
                 fontsize=15, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path.name}")
        
        plt.close()
    
    def plot_residuals(self, save_path: Path = None):
        """
        Plot residual analysis.
        
        Args:
            save_path: Path to save plot
        """
        residuals = self.y_val_original - self.y_pred_original
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residual scatter plot
        ax1 = axes[0]
        ax1.scatter(self.y_pred_original, residuals, alpha=0.5, s=30, 
                   edgecolors='k', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Price (lakhs)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Residuals (lakhs)', fontsize=13, fontweight='bold')
        ax1.set_title(f'Residual Plot\nRMSE = {self.metrics["RMSE"]:.4f}', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residual distribution
        ax2 = axes[1]
        ax2.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        ax2.axvline(x=residuals.mean(), color='g', linestyle='--', lw=2, 
                   label=f'Mean = {residuals.mean():.2f}')
        ax2.set_xlabel('Residuals (lakhs)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        ax2.set_title(f'Residual Distribution\nStd = {residuals.std():.2f}', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path.name}")
        
        plt.close()
    
    def plot_error_distribution(self, save_path: Path = None):
        """
        Plot error distribution and statistics.
        
        Args:
            save_path: Path to save plot
        """
        errors = np.abs(self.y_val_original - self.y_pred_original)
        percentage_errors = (errors / self.y_val_original) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute errors
        ax1 = axes[0]
        ax1.hist(errors, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax1.axvline(x=self.metrics['MAE'], color='g', linestyle='--', lw=2,
                   label=f'MAE = {self.metrics["MAE"]:.2f}')
        ax1.set_xlabel('Absolute Error (lakhs)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        ax1.set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Percentage errors
        ax2 = axes[1]
        ax2.hist(percentage_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axvline(x=self.metrics['MAPE'], color='r', linestyle='--', lw=2,
                   label=f'MAPE = {self.metrics["MAPE"]:.2f}%')
        ax2.set_xlabel('Percentage Error (%)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold')
        ax2.set_title('Percentage Error Distribution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path.name}")
        
        plt.close()
    
    def plot_price_ranges(self, save_path: Path = None):
        """
        Plot performance by price range.
        
        Args:
            save_path: Path to save plot
        """
        # Create price bins
        bins = [0, 50, 100, 150, 200, 300, float('inf')]
        labels = ['<50', '50-100', '100-150', '150-200', '200-300', '>300']
        
        df = pd.DataFrame({
            'Actual': self.y_val_original,
            'Predicted': self.y_pred_original,
            'Error': np.abs(self.y_val_original - self.y_pred_original),
            'Pct_Error': np.abs((self.y_val_original - self.y_pred_original) / self.y_val_original) * 100
        })
        
        df['Price_Range'] = pd.cut(df['Actual'], bins=bins, labels=labels)
        
        # Calculate metrics by range
        range_stats = df.groupby('Price_Range', observed=True).agg({
            'Error': 'mean',
            'Pct_Error': 'mean',
            'Actual': 'count'
        }).reset_index()
        range_stats.columns = ['Price_Range', 'MAE', 'MAPE', 'Count']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # MAE by price range
        ax1 = axes[0]
        bars1 = ax1.bar(range_stats['Price_Range'], range_stats['MAE'], 
                       alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Price Range (lakhs)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Error (lakhs)', fontsize=13, fontweight='bold')
        ax1.set_title('MAE by Price Range', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar, count in zip(bars1, range_stats['Count']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}', ha='center', va='bottom', fontsize=10)
        
        # MAPE by price range
        ax2 = axes[1]
        bars2 = ax2.bar(range_stats['Price_Range'], range_stats['MAPE'], 
                       alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Price Range (lakhs)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=13, fontweight='bold')
        ax2.set_title('MAPE by Price Range', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar, count in zip(bars2, range_stats['Count']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path.name}")
        
        plt.close()
    
    def analyze_worst_predictions(self, n: int = 10) -> pd.DataFrame:
        """
        Analyze worst predictions.
        
        Args:
            n: Number of worst predictions to analyze
            
        Returns:
            DataFrame with worst predictions
        """
        errors = np.abs(self.y_val_original - self.predictions)
        worst_idx = errors.nlargest(n).index
        
        worst_df = pd.DataFrame({
            'Actual_Price': self.y_val_original.loc[worst_idx],
            'Predicted_Price': self.predictions.loc[worst_idx],
            'Absolute_Error': errors.loc[worst_idx],
            'Percentage_Error': (errors.loc[worst_idx] / self.y_val_original.loc[worst_idx] * 100)
        })
        
        # Add features if available
        if hasattr(self, 'X_val') and self.X_val is not None:
            feature_cols = ['SQUARE_FT', 'BHK_NO.', 'city'] if 'city' in self.X_val.columns else ['SQUARE_FT', 'BHK_NO.']
            for col in feature_cols:
                if col in self.X_val.columns:
                    worst_df[col] = self.X_val.loc[worst_idx, col].values
        
        return worst_df
    
    def generate_text_report(self) -> str:
        """
        Generate comprehensive text report.
        
        Returns:
            Report string
        """
        worst_predictions = self.analyze_worst_predictions(5)
        
        report = f"""
{'='*70}
HOUSE PRICE PREDICTION - COMPREHENSIVE EVALUATION REPORT
{'='*70}

Model: {self.model_name}
{'='*70}

PERFORMANCE METRICS
{'='*70}
RMSE (Root Mean Squared Error):    {self.metrics['RMSE']:.4f} lakhs
MAE (Mean Absolute Error):          {self.metrics['MAE']:.4f} lakhs
R² (Coefficient of Determination):  {self.metrics['R2']:.4f}
MAPE (Mean Absolute % Error):       {self.metrics['MAPE']:.2f}%

Negative Predictions:               {self.metrics['Negative_Count']} ({self.metrics['Negative_Pct']:.2f}%)

{'='*70}
INTERPRETATION
{'='*70}
• Average prediction error: {self.metrics['MAE']:.2f} lakhs
• Model explains {self.metrics['R2']*100:.1f}% of price variance
• Typical prediction within {self.metrics['MAPE']:.1f}% of actual price

{'='*70}
DATA SUMMARY
{'='*70}
Validation samples:      {len(self.y_val_original)}
Price range:            {self.y_val_original.min():.2f} - {self.y_val_original.max():.2f} lakhs
Mean actual price:      {self.y_val_original.mean():.2f} lakhs
Median actual price:    {self.y_val_original.median():.2f} lakhs

{'='*70}
WORST PREDICTIONS (Top 5)
{'='*70}
{worst_predictions.to_string(index=False, float_format='%.2f')}

{'='*70}
MODEL STATUS
{'='*70}
"""
        
        if self.metrics['Negative_Pct'] == 0:
            report += "✓ PRODUCTION READY - No negative predictions\n"
        else:
            report += f"⚠️  REVIEW NEEDED - {self.metrics['Negative_Pct']:.2f}% negative predictions\n"
        
        if self.metrics['MAPE'] < 40:
            report += "✓ MAPE within industry acceptable range (<40%)\n"
        else:
            report += f"⚠️  MAPE above industry standard (>40%)\n"
        
        report += f"\n{'='*70}\n"
        report += "Report generated using log-transformed target\n" if config.USE_LOG_TRANSFORM else "Report generated without log transformation\n"
        report += f"{'='*70}\n"
        
        return report
    
    def save_metrics(self, filepath: Path):
        """
        Save metrics to CSV.
        
        Args:
            filepath: Path to save metrics
        """
        df = pd.DataFrame([self.metrics])
        df.to_csv(filepath, index=False)
        print(f"✓ Saved metrics: {filepath.name}")
    
    def generate_all_visualizations(self):
        """Generate and save all visualizations."""
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}\n")
        
        # Calculate metrics first
        self.calculate_metrics()
        
        # Generate plots
        self.plot_predictions_vs_actual(config.PLOTS_DIR / "01_predictions_vs_actual.png")
        self.plot_residuals(config.PLOTS_DIR / "02_residuals.png")
        self.plot_error_distribution(config.PLOTS_DIR / "03_error_distribution.png")
        self.plot_price_ranges(config.PLOTS_DIR / "04_performance_by_price_range.png")
        
        print(f"\n✓ All plots saved to: {config.PLOTS_DIR}")
    
    def generate_full_report(self):
        """Generate complete evaluation: metrics, plots, and text report."""
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE EVALUATION")
        print(f"{'='*60}\n")
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Print metrics
        print("PERFORMANCE METRICS")
        print("-" * 60)
        print(f"  RMSE:     {self.metrics['RMSE']:.4f} lakhs")
        print(f"  MAE:      {self.metrics['MAE']:.4f} lakhs")
        print(f"  R²:       {self.metrics['R2']:.4f}")
        print(f"  MAPE:     {self.metrics['MAPE']:.2f}%")
        print(f"  Negative: {self.metrics['Negative_Count']} ({self.metrics['Negative_Pct']:.2f}%)")
        if self.metrics['Negative_Pct'] == 0:
            print("  ✓ No negative predictions!")
        print()
        
        # Generate visualizations
        self.generate_all_visualizations()
        
        # Save metrics
        self.save_metrics(config.METRICS_PATH)
        
        # Generate and save text report
        report = self.generate_text_report()
        with open(config.REPORT_PATH, 'w') as f:
            f.write(report)
        print(f"✓ Saved report: {config.REPORT_PATH.name}")
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Plots:   {config.PLOTS_DIR}")
        print(f"  Metrics: {config.METRICS_PATH}")
        print(f"  Report:  {config.REPORT_PATH}")
        print(f"{'='*60}\n")
        
        return self.metrics


def main():
    """Example usage of ModelEvaluator."""
    # This would be called after model training
    print("ModelEvaluator is ready for use in the pipeline")
    print("See main.py for integration example")


if __name__ == "__main__":
    main()