"""
Label Generation Module for Stock Analysis

This module provides functions to generate labels for stock trading data
based on quarterly excess returns. It supports multiple labeling strategies:
- 5-class classification (Lognormal distribution)
- 3-class classification (Lognormal distribution)
- Regression labels (continuous returns)

Usage:
    from generate_labels import LabelGenerator
    
    gen = LabelGenerator(stock_list_path="Eastmoney_report_pdf_download/HS300.csv")
    all_returns = gen.load_and_aggregate_returns(trading_data_folder="trading_data")
    quarterly_excess = gen.calculate_excess_returns(all_returns)
    
    # Generate labels using different strategies
    labels_5class = gen.generate_5class_labels(quarterly_excess)
    labels_3class = gen.generate_3class_labels(quarterly_excess)
    labels_regression = gen.generate_regression_labels(quarterly_excess)
    
    # Save labels
    gen.save_labels(labels_5class, labels_3class, labels_regression)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from typing import Optional, Dict, Tuple


class LabelGenerator:
    """
    Class to generate classification and regression labels for stock trading data.
    """

    def __init__(self, stock_list_path: str = "Eastmoney_report_pdf_download/HS300.csv"):
        """
        Initialize the label generator.
        
        Args:
            stock_list_path: Path to CSV file containing stock list with '股票代码' column
        """
        self.stock_list_path = stock_list_path
        self.stock_list = None
        self.all_returns = None
        self.quarterly_excess_returns = None

    def load_stock_list(self) -> pd.DataFrame:
        """
        Load stock list from CSV file.
        
        Returns:
            DataFrame containing stock codes from the CSV file
        """
        self.stock_list = pd.read_csv(self.stock_list_path, encoding='utf-8')
        return self.stock_list

    @staticmethod
    def process_single_stock(code, folder: str = "trading_data") -> Optional[pd.DataFrame]:
        """
        Process a single stock's CSV file and return as DataFrame.
        
        Args:
            code: Stock code (string or int), or '沪深300' for index
            folder: Path to folder containing trading data CSV files
            
        Returns:
            DataFrame with trading date as index and price changes as values,
            or None if file not found
        """
        if code == '沪深300':
            codes_padded = code
        else:
            codes_padded = str(code).zfill(6)  # Pad to 6 digits
        
        path = Path(folder) / f"{codes_padded}.csv"

        if not path.exists():
            print(f"File not found: {path}")
            return None

        temp = pd.read_csv(path, encoding='utf-8')
        temp = temp[['交易日期', '涨跌幅(%)']]
        # Remove duplicates based on trading date
        temp = temp.drop_duplicates(subset=['交易日期'])
        # Rename return percentage column to stock code
        temp = temp.rename(columns={'涨跌幅(%)': codes_padded})
        temp = temp.set_index('交易日期')

        return temp

    def load_and_aggregate_returns(self, trading_data_folder: str = "trading_data",
                                   output_csv: str = "all_stock_returns.csv") -> pd.DataFrame:
        """
        Load trading data for all stocks and aggregate into a single DataFrame.
        
        Args:
            trading_data_folder: Path to folder containing individual stock CSV files
            output_csv: Path to save the aggregated returns CSV
            
        Returns:
            DataFrame with aggregated daily returns for all stocks
        """
        if self.stock_list is None:
            self.load_stock_list()

        all_data = []

        # Load data for each stock in the list
        for code in self.stock_list['股票代码']:
            df = self.process_single_stock(code, folder=trading_data_folder)
            if df is not None:
                all_data.append(df)

        # Add index data (HS300)
        all_data.append(self.process_single_stock('沪深300', folder=trading_data_folder))

        # Concatenate all data
        big_table = pd.concat(all_data, axis=1, join='outer').reset_index().sort_values(
            by='交易日期'
        )
        big_table['交易日期'] = pd.to_datetime(big_table['交易日期'], format='%Y%m%d')
        big_table = big_table.dropna(subset=['交易日期'])
        
        # Save to CSV
        big_table.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Saved aggregated returns to {output_csv}")
        
        self.all_returns = big_table
        return big_table

    def convert_to_quarterly_returns(self, daily_returns: Optional[pd.DataFrame] = None,
                                     output_csv: str = "quarterly_returns.csv") -> pd.DataFrame:
        """
        Convert daily returns to quarterly returns using compound growth.
        
        Args:
            daily_returns: DataFrame with daily returns. If None, uses self.all_returns
            output_csv: Path to save quarterly returns CSV
            
        Returns:
            DataFrame with quarterly compound returns
        """
        if daily_returns is None:
            if self.all_returns is None:
                raise ValueError("No returns data provided. Run load_and_aggregate_returns() first.")
            daily_returns = self.all_returns

        # Convert all return columns to numeric and handle errors
        returns_data = daily_returns.drop(columns=['交易日期']).copy()
        for col in returns_data.columns:
            returns_data[col] = pd.to_numeric(returns_data[col], errors='coerce')
        
        # Convert return percentages to decimals
        returns = returns_data / 100.0

        # Calculate compound factors (1 + return)
        factors = 1 + returns
        factors['交易日期'] = daily_returns['交易日期']

        # Convert to quarter periods
        factors['交易日期'] = pd.to_datetime(factors['交易日期']).dt.to_period("Q")

        # Aggregate by quarter using product and convert back to returns
        quarterly_returns = (factors.groupby('交易日期').prod() - 1).reset_index()
        
        # Save to CSV
        quarterly_returns.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Saved quarterly returns to {output_csv}")
        
        return quarterly_returns

    def calculate_excess_returns(self, quarterly_returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate excess returns by subtracting benchmark (HS300) returns.
        
        Args:
            quarterly_returns: Quarterly returns DataFrame. If None, generates from daily returns
            
        Returns:
            DataFrame with excess returns and trading dates
        """
        if quarterly_returns is None:
            if self.all_returns is None:
                raise ValueError("No returns data provided. Run load_and_aggregate_returns() first.")
            quarterly_returns = self.convert_to_quarterly_returns(self.all_returns)

        # Calculate excess returns (stock return - benchmark return)
        quarterly_excess_returns = quarterly_returns.drop(columns=['交易日期']).sub(
            quarterly_returns['沪深300'], axis=0
        )
        quarterly_excess_returns['交易日期'] = quarterly_returns['交易日期']
        
        self.quarterly_excess_returns = quarterly_excess_returns
        return quarterly_excess_returns

    @staticmethod
    def analyze_return_distribution(excess_returns: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of excess returns using multiple statistical tests.
        
        Args:
            excess_returns: DataFrame with excess returns
            
        Returns:
            Dictionary containing distribution analysis results
        """
        numeric_df = excess_returns.select_dtypes(include=[np.number])

        # Flatten and clean data
        flattened = numeric_df.values.flatten()
        flattened = flattened[~np.isnan(flattened)]

        # Convert to total return factors
        total_return_factor = 1 + flattened
        total_return_factor = total_return_factor[total_return_factor > 0]

        # Fit different distributions
        distributions = {
            "Normal": stats.norm,
            "Lognormal": stats.lognorm,
            "t": stats.t,
            "Laplace": stats.laplace
        }

        fit_results = []
        for name, dist in distributions.items():
            try:
                # Use log returns for Lognormal, raw returns for others
                data_to_fit = flattened if name != "Lognormal" else total_return_factor
                params = dist.fit(data_to_fit)
                ks_stat, ks_p = stats.kstest(data_to_fit, dist.name, args=params)
                fit_results.append({
                    "Distribution": name,
                    "Params": params,
                    "KS_stat": ks_stat,
                    "KS_p_value": ks_p
                })
            except Exception as e:
                fit_results.append({
                    "Distribution": name,
                    "Params": str(e),
                    "KS_stat": None,
                    "KS_p_value": None
                })

        results_df = pd.DataFrame(fit_results)
        print("Distribution Analysis Results:")
        print(results_df)
        
        return {
            "results": results_df,
            "flattened_returns": flattened,
            "total_return_factors": total_return_factor
        }

    @staticmethod
    def plot_return_distribution(excess_returns: pd.DataFrame):
        """
        Plot histogram, KDE, and QQ plot of excess returns.
        
        Args:
            excess_returns: DataFrame with excess returns
        """
        numeric_df = excess_returns.select_dtypes(include=[np.number])
        flattened = numeric_df.values.flatten()
        flattened = flattened[~np.isnan(flattened)]
        total_return_factor = 1 + flattened
        total_return_factor = total_return_factor[total_return_factor > 0]

        # Histogram with KDE
        plt.figure(figsize=(10, 6))
        sns.histplot(total_return_factor, bins=50, kde=True, stat="density", color="skyblue")
        plt.title("Quarterly Excess Return Factors Distribution (1+R)")
        plt.xlabel("Total Return Factor")
        plt.ylabel("Density")
        plt.show()

        # QQ plot
        plt.figure(figsize=(6, 6))
        stats.probplot(np.log(total_return_factor), dist="norm", plot=plt)
        plt.title("QQ Plot of log(1+R) vs Normal")
        plt.show()

    @staticmethod
    def generate_5class_labels(excess_returns: pd.DataFrame, eps: float = 1e-6,
                               lower_pct: float = 0.1, upper_pct: float = 99.9) -> pd.DataFrame:
        """
        Generate 5-class labels using Lognormal distribution quantiles.
        Classes: 0 (very negative), 1 (negative), 2 (neutral), 3 (positive), 4 (very positive)
        
        Args:
            excess_returns: DataFrame with excess returns grouped by quarter
            eps: Threshold for near-zero returns (treated as neutral)
            lower_pct: Lower percentile for winsorization
            upper_pct: Upper percentile for winsorization
            
        Returns:
            DataFrame with 5-class labels and trading dates
        """
        labels = []

        for quarter, g in excess_returns.groupby("交易日期"):
            numeric_df = g.select_dtypes(include=[np.number])
            flattened = numeric_df.values.flatten()
            flattened = flattened[~np.isnan(flattened)]
            # Ignore near-zero returns
            flattened = flattened[np.abs(flattened) > eps]

            # Convert to total return factors (must be positive)
            total_return = 1 + flattened
            total_return = total_return[total_return > 0]

            # If insufficient samples, set to neutral
            if len(total_return) < 50:
                g_labels = pd.DataFrame(2, index=g.index, columns=numeric_df.columns)
            else:
                # Winsorize to remove extreme values
                lower, upper = np.percentile(total_return, [lower_pct, upper_pct])
                total_return = np.clip(total_return, lower, upper)

                # Fit lognormal distribution (log returns approximate normal)
                log_ret = np.log(total_return)

                # Calculate quantile thresholds
                q20, q40, q60, q80 = np.percentile(log_ret, [20, 40, 60, 80])

                def label_5class(x):
                    if pd.isna(x):
                        return np.nan
                    elif abs(x) <= eps:
                        return 2  # Non-tradable → Neutral
                    r = 1 + x
                    if r <= 0:
                        return 2  # Invalid value → Neutral
                    log_r = np.log(r)
                    if log_r < q20:
                        return 0
                    elif log_r < q40:
                        return 1
                    elif log_r < q60:
                        return 2
                    elif log_r < q80:
                        return 3
                    else:
                        return 4

                g_labels = numeric_df.map(label_5class)

            g_labels["交易日期"] = g["交易日期"]
            labels.append(g_labels)

        return pd.concat(labels)

    @staticmethod
    def generate_3class_labels(excess_returns: pd.DataFrame, eps: float = 1e-6,
                               lower_pct: float = 0.1, upper_pct: float = 99.9) -> pd.DataFrame:
        """
        Generate 3-class labels using Lognormal distribution quantiles.
        Classes: 0 (negative), 1 (neutral), 2 (positive)
        
        Args:
            excess_returns: DataFrame with excess returns grouped by quarter
            eps: Threshold for near-zero returns (treated as neutral)
            lower_pct: Lower percentile for winsorization
            upper_pct: Upper percentile for winsorization
            
        Returns:
            DataFrame with 3-class labels and trading dates
        """
        labels = []

        for quarter, g in excess_returns.groupby("交易日期"):
            numeric_df = g.select_dtypes(include=[np.number])
            flattened = numeric_df.values.flatten()
            flattened = flattened[~np.isnan(flattened)]
            # Ignore near-zero returns
            flattened = flattened[np.abs(flattened) > eps]

            # Convert to total return factors (must be positive)
            total_return = 1 + flattened
            total_return = total_return[total_return > 0]

            # If insufficient samples, set to neutral
            if len(total_return) < 50:
                g_labels = pd.DataFrame(1, index=g.index, columns=numeric_df.columns)
            else:
                # Winsorize to remove extreme values
                lower, upper = np.percentile(total_return, [lower_pct, upper_pct])
                total_return = np.clip(total_return, lower, upper)

                # Fit lognormal distribution (log returns approximate normal)
                log_ret = np.log(total_return)

                # Calculate quantile thresholds for 3 classes
                q33, q67 = np.percentile(log_ret, [33, 67])

                def label_3class(x):
                    if pd.isna(x):
                        return np.nan
                    elif abs(x) <= eps:
                        return 1  # Non-tradable → Neutral
                    r = 1 + x
                    if r <= 0:
                        return 1  # Invalid value → Neutral
                    log_r = np.log(r)
                    if log_r < q33:
                        return 0  # Negative class
                    elif log_r < q67:
                        return 1  # Neutral class
                    else:
                        return 2  # Positive class

                g_labels = numeric_df.map(label_3class)

            g_labels["交易日期"] = g["交易日期"]
            labels.append(g_labels)

        return pd.concat(labels)

    @staticmethod
    def generate_regression_labels(excess_returns: pd.DataFrame, eps: float = 1e-6,
                                   lower_pct: float = 0.1, upper_pct: float = 99.9) -> pd.DataFrame:
        """
        Generate continuous regression labels based on excess returns.
        
        Args:
            excess_returns: DataFrame with excess returns grouped by quarter
            eps: Threshold for near-zero returns
            lower_pct: Lower percentile for winsorization
            upper_pct: Upper percentile for winsorization
            
        Returns:
            DataFrame with continuous regression labels and trading dates
        """
        labels = []

        for quarter, g in excess_returns.groupby("交易日期"):
            numeric_df = g.select_dtypes(include=[np.number])
            flattened = numeric_df.values.flatten()
            flattened = flattened[~np.isnan(flattened)]
            # Ignore near-zero returns
            flattened = flattened[np.abs(flattened) > eps]

            # Convert to total return factors (must be positive)
            total_return = 1 + flattened
            total_return = total_return[total_return > 0]

            # If insufficient samples, return NaN
            if len(total_return) < 50:
                g_labels = pd.DataFrame(np.nan, index=g.index, columns=numeric_df.columns)
            else:
                # Winsorize to remove extreme values
                lower, upper = np.percentile(total_return, [lower_pct, upper_pct])
                # Keep original excess return scale
                clipped = np.clip(1 + numeric_df, lower, upper) - 1

                # Use directly as regression labels
                g_labels = clipped

            g_labels["交易日期"] = g["交易日期"]
            labels.append(g_labels)

        return pd.concat(labels)

    @staticmethod
    def get_label_statistics(labels: pd.DataFrame, label_type: str = "5class") -> pd.DataFrame:
        """
        Calculate statistics for label distribution.
        
        Args:
            labels: DataFrame containing label values
            label_type: Type of labels ('5class', '3class', or 'regression')
            
        Returns:
            DataFrame showing label percentages/statistics
        """
        label_counts = labels.drop(columns=['交易日期']).stack().value_counts(normalize=True)
        label_percentages = (label_counts * 100).round(2)
        
        print(f"\nLabel Distribution ({label_type}):")
        print(label_percentages)
        
        return label_percentages

    @staticmethod
    def save_labels(labels_5class: Optional[pd.DataFrame] = None,
                    labels_3class: Optional[pd.DataFrame] = None,
                    labels_regression: Optional[pd.DataFrame] = None,
                    output_dir: str = "."):
        """
        Save labels to CSV files.
        
        Args:
            labels_5class: 5-class labels DataFrame
            labels_3class: 3-class labels DataFrame
            labels_regression: Regression labels DataFrame
            output_dir: Directory to save output files
        """
        if labels_5class is not None:
            labels_5class.to_csv(
                f"{output_dir}/labels_5class.csv", index=False, encoding="utf-8"
            )
            print(f"Saved 5-class labels to {output_dir}/labels_5class.csv")

        if labels_3class is not None:
            labels_3class.to_csv(
                f"{output_dir}/labels_3class.csv", index=False, encoding="utf-8"
            )
            print(f"Saved 3-class labels to {output_dir}/labels_3class.csv")

        if labels_regression is not None:
            labels_regression.to_csv(
                f"{output_dir}/labels_regression.csv", index=False, encoding="utf-8"
            )
            print(f"Saved regression labels to {output_dir}/labels_regression.csv")


def main():
    """
    Main execution function demonstrating the complete workflow.
    """
    # Initialize label generator
    gen = LabelGenerator(
        stock_list_path="Eastmoney_report_pdf_download/HS300.csv"
    )

    # Step 1: Load stock list
    print("Loading stock list...")
    gen.load_stock_list()

    # Step 2: Load and aggregate returns
    print("Loading and aggregating trading data...")
    all_returns = gen.load_and_aggregate_returns(trading_data_folder="trading_data")

    # Step 3: Calculate quarterly excess returns
    print("Calculating quarterly excess returns...")
    quarterly_excess = gen.calculate_excess_returns()

    # Step 4: Analyze distribution
    print("\nAnalyzing return distribution...")
    gen.plot_return_distribution(quarterly_excess)
    gen.analyze_return_distribution(quarterly_excess)

    # Step 5: Generate labels
    print("\nGenerating labels...")
    labels_5class = gen.generate_5class_labels(quarterly_excess)
    labels_3class = gen.generate_3class_labels(quarterly_excess)
    labels_regression = gen.generate_regression_labels(quarterly_excess)

    # Step 6: Get statistics
    print("\n" + "="*50)
    gen.get_label_statistics(labels_5class, "5-class")
    gen.get_label_statistics(labels_3class, "3-class")
    print("="*50)

    # Step 7: Save labels
    gen.save_labels(labels_5class, labels_3class, labels_regression)

    print("\nLabel generation complete!")


if __name__ == "__main__":
    main()
