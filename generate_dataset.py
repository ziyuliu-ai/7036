"""
Dataset Generation Module for Stock Sentiment Analysis

This module provides reusable functions to generate training, validation, and 
backtesting datasets from cleaned English reports and corresponding labels.

Features:
- Load and map stock data
- Process reports by quarter
- Support for 3-class, 5-class, and regression labels
- Time-series aware dataset splitting
- Flexible train/validation/holdout splits

Usage Example:
    from generate_dataset import DatasetGenerator
    
    gen = DatasetGenerator(
        labels_file="labels_5class.csv",
        mapping_file="Eastmoney_report_pdf_download/HS300.csv",
        reports_dir="reports_txt_by_quarter_cleaned_en"
    )
    
    # Generate 5-class dataset with stratified split
    train_df, val_df = gen.generate_5class_dataset()
    
    # Generate timeseries 3-class dataset
    train_df, val_df = gen.generate_timeseries_3class_dataset()
    
    # Generate regression dataset with holdout set
    train_df, val_df, test_df = gen.generate_regression_dataset()
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List


class DatasetGenerator:
    """
    Main class for generating datasets from reports and labels.
    
    Attributes:
        labels_file (str): Path to labels CSV file
        mapping_file (str): Path to stock mapping CSV file
        reports_dir (str): Path to reports directory
        name2code (Dict[str, str]): Company name to stock code mapping
        min_text_length (int): Minimum report text length threshold
    """
    
    def __init__(
        self, 
        labels_file: str,
        mapping_file: str = "Eastmoney_report_pdf_download/HS300.csv",
        reports_dir: str = "reports_txt_by_quarter_cleaned_en",
        min_text_length: int = 50
    ):
        """
        Initialize DatasetGenerator.
        
        Args:
            labels_file: Path to labels CSV file
            mapping_file: Path to stock mapping CSV file
            reports_dir: Path to reports directory
            min_text_length: Minimum report text length to include
        """
        self.labels_file = labels_file
        self.mapping_file = mapping_file
        self.reports_dir = reports_dir
        self.min_text_length = min_text_length
        
        # Load mapping
        self.name2code = self._load_mapping()
    
    def _load_mapping(self) -> Dict[str, str]:
        """
        Load company name to stock code mapping.
        
        Returns:
            Dictionary mapping company names to stock codes (zero-padded to 6 digits)
        """
        mapping = pd.read_csv(self.mapping_file, dtype={"股票代码": str})
        mapping["股票代码"] = mapping["股票代码"].str.zfill(6)
        return dict(zip(mapping["股票简称"], mapping["股票代码"]))
    
    def _collect_reports(
        self, 
        label_type: str = "class",
        preserve_quarter: bool = False
    ) -> List[Dict]:
        """
        Collect all reports and corresponding labels.
        
        Args:
            label_type: "class" for classification labels, "regression" for continuous
            preserve_quarter: Whether to preserve quarter information in output
        
        Returns:
            List of dictionaries with "text", "label", and optionally "quarter"
        """
        labels = pd.read_csv(self.labels_file)
        data = []
        
        quarter_dirs = sorted(os.listdir(self.reports_dir))
        
        for quarter in quarter_dirs:
            quarter_path = os.path.join(self.reports_dir, quarter)
            if not os.path.isdir(quarter_path):
                continue
            
            # Convert folder name (2017_Q1) to label key format (2017Q1)
            quarter_key = quarter.replace("_", "")
            label_row = labels[labels["交易日期"] == quarter_key]
            if label_row.empty:
                continue
            
            for company_name in os.listdir(quarter_path):
                company_path = os.path.join(quarter_path, company_name)
                if not os.path.isdir(company_path):
                    continue
                
                # Map company name to stock code
                if company_name not in self.name2code:
                    continue
                stock_code = self.name2code[company_name]
                
                # Get label for this company in this quarter
                if stock_code not in label_row.columns:
                    continue
                
                label_value = label_row[stock_code].values[0]
                if label_type == "regression":
                    company_label = float(label_value)
                else:
                    company_label = int(label_value)
                
                # Collect all report files
                for root, _, files in os.walk(company_path):
                    for file in files:
                        if file.endswith(".txt"):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    text = f.read().strip()
                                
                                if len(text) > self.min_text_length:
                                    record = {"text": text, "label": company_label}
                                    if preserve_quarter:
                                        record["quarter"] = quarter_key
                                    if label_type == "regression":
                                        record["stock_code"] = stock_code
                                    data.append(record)
                            except Exception as e:
                                print(f"Skip file {file_path}: {e}")
        
        return data
    
    def generate_5class_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate 5-class dataset with stratified train/validation split.
        
        Returns:
            Tuple of (train_df, val_df)
            - Split: 80/20
            - Each split: train (60%), validation (20%), dropped to ensure stratification
        """
        data = self._collect_reports(label_type="class", preserve_quarter=False)
        df = pd.DataFrame(data)
        
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df["label"]
        )
        
        train_df.to_csv("train_5class.csv", index=False)
        val_df.to_csv("val_5class.csv", index=False)
        
        print(f"✓ Generated: train_5class.csv ({len(train_df)} samples), "
              f"val_5class.csv ({len(val_df)} samples)")
        
        return train_df, val_df
    
    def generate_3class_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate 3-class dataset with stratified train/validation split.
        
        Returns:
            Tuple of (train_df, val_df)
            - Split: 80/20
        """
        data = self._collect_reports(label_type="class", preserve_quarter=False)
        df = pd.DataFrame(data)
        
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df["label"]
        )
        
        train_df.to_csv("train_3class.csv", index=False)
        val_df.to_csv("val_3class.csv", index=False)
        
        print(f"✓ Generated: train_3class.csv ({len(train_df)} samples), "
              f"val_3class.csv ({len(val_df)} samples)")
        
        return train_df, val_df
    
    def generate_timeseries_5class_dataset(
        self,
        val_start: str = "2024Q4",
        val_end: str = "2025Q4"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate 5-class timeseries dataset with fixed validation period.
        
        Args:
            val_start: Validation period start (e.g., "2024Q4")
            val_end: Validation period end (e.g., "2025Q4")
        
        Returns:
            Tuple of (train_df, val_df)
            - Training: Before val_start
            - Validation: From val_start to val_end (inclusive)
        """
        data = self._collect_reports(label_type="class", preserve_quarter=True)
        df = pd.DataFrame(data)
        df = df.sort_values("quarter")
        
        # Filter quarters in validation range
        val_quarters = [q for q in df["quarter"].unique() 
                       if val_start <= q <= val_end]
        
        train_df = df[~df["quarter"].isin(val_quarters)]
        val_df = df[df["quarter"].isin(val_quarters)]
        
        train_df.to_csv("train_timeseries_5class.csv", index=False)
        val_df.to_csv("val_timeseries_5class.csv", index=False)
        
        print(f"✓ Generated timeseries 5-class: "
              f"train ({len(train_df)} samples, {train_df['quarter'].min()}–{train_df['quarter'].max()}), "
              f"val ({len(val_df)} samples, {val_df['quarter'].min()}–{val_df['quarter'].max()})")
        
        return train_df, val_df
    
    def generate_timeseries_3class_dataset(
        self,
        val_start: str = "2024Q4",
        val_end: str = "2025Q4"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate 3-class timeseries dataset with fixed validation period.
        
        Args:
            val_start: Validation period start (e.g., "2024Q4")
            val_end: Validation period end (e.g., "2025Q4")
        
        Returns:
            Tuple of (train_df, val_df)
        """
        data = self._collect_reports(label_type="class", preserve_quarter=True)
        df = pd.DataFrame(data)
        df = df.sort_values("quarter")
        
        val_quarters = [q for q in df["quarter"].unique() 
                       if val_start <= q <= val_end]
        
        train_df = df[~df["quarter"].isin(val_quarters)]
        val_df = df[df["quarter"].isin(val_quarters)]
        
        train_df.to_csv("train_timeseries_3class.csv", index=False)
        val_df.to_csv("val_timeseries_3class.csv", index=False)
        
        print(f"✓ Generated timeseries 3-class: "
              f"train ({len(train_df)} samples, {train_df['quarter'].min()}–{train_df['quarter'].max()}), "
              f"val ({len(val_df)} samples, {val_df['quarter'].min()}–{val_df['quarter'].max()})")
        
        return train_df, val_df
    
    def generate_regression_dataset(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.1,
        test_ratio: float = 0.3,
        stratify_by_quarter: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate regression dataset with train/val/test split.
        
        Args:
            train_ratio: Proportion for training (default: 0.6 = 60%)
            val_ratio: Proportion for validation (default: 0.1 = 10%)
            test_ratio: Proportion for holdout/backtest (default: 0.3 = 30%)
            stratify_by_quarter: Whether to stratify splits by quarter
        
        Returns:
            Tuple of (train_df, val_df, test_df)
            - Each split contains: text, label, quarter, stock_code columns
        """
        data = self._collect_reports(label_type="regression", preserve_quarter=True)
        df = pd.DataFrame(data)
        
        if not stratify_by_quarter:
            # Simple split
            temp_train, test_df = train_test_split(
                df, test_size=test_ratio, random_state=42
            )
            train_df, val_df = train_test_split(
                temp_train, test_size=val_ratio/(1-test_ratio), random_state=42
            )
        else:
            # Stratified by quarter to preserve temporal distribution
            train_list, val_list, test_list = [], [], []
            
            for quarter, group in df.groupby("quarter"):
                # First split: train+val vs test
                temp_trainval, temp_test = train_test_split(
                    group, test_size=test_ratio, random_state=42
                )
                # Second split: train vs val
                temp_train, temp_val = train_test_split(
                    temp_trainval,
                    test_size=val_ratio/(1-test_ratio),
                    random_state=42
                )
                
                train_list.append(temp_train)
                val_list.append(temp_val)
                test_list.append(temp_test)
            
            train_df = pd.concat(train_list).reset_index(drop=True)
            val_df = pd.concat(val_list).reset_index(drop=True)
            test_df = pd.concat(test_list).reset_index(drop=True)
        
        train_df.to_csv("train_regression.csv", index=False)
        val_df.to_csv("val_regression.csv", index=False)
        test_df.to_csv("backtest_regression.csv", index=False)
        
        print(f"✓ Generated regression dataset: "
              f"train ({len(train_df)} samples), "
              f"val ({len(val_df)} samples), "
              f"test ({len(test_df)} samples)")
        
        return train_df, val_df, test_df
    
    def generate_timeseries_regression_dataset(
        self,
        val_start: str = "2024Q4",
        val_end: str = "2025Q4"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate regression timeseries dataset with fixed validation period.
        
        Args:
            val_start: Validation period start (e.g., "2024Q4")
            val_end: Validation period end (e.g., "2025Q4")
        
        Returns:
            Tuple of (train_df, val_df)
        """
        data = self._collect_reports(label_type="regression", preserve_quarter=True)
        df = pd.DataFrame(data)
        df = df.sort_values("quarter")
        
        val_quarters = [q for q in df["quarter"].unique() 
                       if val_start <= q <= val_end]
        
        train_df = df[~df["quarter"].isin(val_quarters)]
        val_df = df[df["quarter"].isin(val_quarters)]
        
        train_df.to_csv("train_timeseries_regression.csv", index=False)
        val_df.to_csv("val_timeseries_regression.csv", index=False)
        
        print(f"✓ Generated timeseries regression: "
              f"train ({len(train_df)} samples, {train_df['quarter'].min()}–{train_df['quarter'].max()}), "
              f"val ({len(val_df)} samples, {val_df['quarter'].min()}–{val_df['quarter'].max()})")
        
        return train_df, val_df


def generate_all_datasets(
    labels_5class: str = "labels_5class.csv",
    labels_3class: str = "labels_3class.csv",
    labels_regression: str = "labels_regression.csv",
    mapping_file: str = "Eastmoney_report_pdf_download/HS300.csv",
    reports_dir: str = "reports_txt_by_quarter_cleaned_en"
) -> Dict[str, Tuple]:
    """
    Convenience function to generate all dataset types at once.
    
    Args:
        labels_5class: Path to 5-class labels file
        labels_3class: Path to 3-class labels file
        labels_regression: Path to regression labels file
        mapping_file: Path to stock mapping file
        reports_dir: Path to reports directory
    
    Returns:
        Dictionary with all generated datasets and their DataFrames
        
    Example:
        datasets = generate_all_datasets()
        # Access with:
        # datasets['train_5class_df'], datasets['val_5class_df']
        # datasets['train_regression_df'], datasets['val_regression_df'], datasets['test_regression_df']
    """
    results = {}
    
    # 5-class stratified
    print("\n=== Generating 5-class Dataset ===")
    gen_5class = DatasetGenerator(labels_5class, mapping_file, reports_dir)
    train_5, val_5 = gen_5class.generate_5class_dataset()
    results['train_5class_df'] = train_5
    results['val_5class_df'] = val_5
    
    # 3-class stratified
    print("\n=== Generating 3-class Dataset ===")
    gen_3class = DatasetGenerator(labels_3class, mapping_file, reports_dir)
    train_3, val_3 = gen_3class.generate_3class_dataset()
    results['train_3class_df'] = train_3
    results['val_3class_df'] = val_3
    
    # Timeseries 5-class
    print("\n=== Generating Timeseries 5-class Dataset ===")
    train_ts5, val_ts5 = gen_5class.generate_timeseries_5class_dataset()
    results['train_timeseries_5class_df'] = train_ts5
    results['val_timeseries_5class_df'] = val_ts5
    
    # Timeseries 3-class
    print("\n=== Generating Timeseries 3-class Dataset ===")
    train_ts3, val_ts3 = gen_3class.generate_timeseries_3class_dataset()
    results['train_timeseries_3class_df'] = train_ts3
    results['val_timeseries_3class_df'] = val_ts3
    
    # Regression (with holdout)
    print("\n=== Generating Regression Dataset ===")
    gen_reg = DatasetGenerator(labels_regression, mapping_file, reports_dir)
    train_reg, val_reg, test_reg = gen_reg.generate_regression_dataset()
    results['train_regression_df'] = train_reg
    results['val_regression_df'] = val_reg
    results['test_regression_df'] = test_reg
    
    # Timeseries Regression
    print("\n=== Generating Timeseries Regression Dataset ===")
    train_ts_reg, val_ts_reg = gen_reg.generate_timeseries_regression_dataset()
    results['train_timeseries_regression_df'] = train_ts_reg
    results['val_timeseries_regression_df'] = val_ts_reg
    
    return results


if __name__ == "__main__":
    """
    Example usage when running directly:
    
    python generate_dataset.py
    """
    
    # Generate all datasets
    print("Starting dataset generation...\n")
    datasets = generate_all_datasets()
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Generated {len(datasets)} datasets")
    print("\nGenerated files:")
    print("  - train_5class.csv, val_5class.csv")
    print("  - train_3class.csv, val_3class.csv")
    print("  - train_timeseries_5class.csv, val_timeseries_5class.csv")
    print("  - train_timeseries_3class.csv, val_timeseries_3class.csv")
    print("  - train_regression.csv, val_regression.csv, backtest_regression.csv")
    print("  - train_timeseries_regression.csv, val_timeseries_regression.csv")
