"""
Backtesting Module for NLP-based Trading Strategy

This module provides complete backtesting functionality for sentiment-based
stock trading strategies, including:
- Strategy return calculation
- Performance metrics (Sharpe ratio, Information ratio)
- Visualization and reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def backtest(return_data, quarter_weights):
    """
    Calculate weighted returns for each quarter using backtesting weights.
    
    Args:
        return_data: DataFrame with quarterly returns by stock
        quarter_weights: DataFrame with weights for each (quarter, stock) pair
    
    Returns:
        Series with weighted returns for each quarter
    """
    return quarter_weights.groupby(by='quarter').apply(
        lambda x: compute_weighted_return(return_data, x)
    )


def compute_weighted_return(return_data, group):
    """
    Compute weighted return for a specific quarter.
    
    Args:
        return_data: DataFrame with quarterly returns
        group: DataFrame subset for a specific quarter
    
    Returns:
        Series with weighted returns
    """
    quarter = group.name
    data = return_data.loc[return_data['quarter'] == quarter, :].copy()
    data.drop('quarter', axis=1, inplace=True)
    group = group.set_index('stock')
    return (group['weight'] * data).sum(axis=1)


def prepare_quarterly_returns(all_stock_returns_file='all_stock_returns.csv'):
    """
    Prepare quarterly returns from daily/all stock returns data.
    
    Args:
        all_stock_returns_file: Path to CSV file with stock returns
    
    Returns:
        DataFrame with quarterly returns
    """
    print("[*] Loading and processing returns data...")
    re_data = pd.read_csv(all_stock_returns_file)
    re_data['交易日期'] = pd.to_datetime(re_data['交易日期'], format='%Y-%m-%d')
    re_data = re_data[re_data['交易日期'] >= '2017-01-01'].reset_index()
    re_data.set_index('交易日期', inplace=True)
    re_data.fillna(0, inplace=True)
    re_data.drop('index', axis=1, inplace=True)
    re_data['沪深300'] = re_data['沪深300'].str.rstrip('%').astype(float)
    
    # Resample to quarterly
    q_data = re_data.resample('Q').apply(lambda x: ((1 + x/100).prod() - 1)*100)
    q_data.index = q_data.index.to_period('Q')
    q_data.rename_axis('quarter', axis=0, inplace=True)
    q_data.reset_index(inplace=True)
    
    print(f"✓ Loaded quarterly returns: {q_data.shape}")
    q_data.to_csv('quarter_return.csv', index=False)
    print(f"✓ Saved to quarter_return.csv")
    
    return q_data


def visualize_returns(avg_return, title_suffix=""):
    """
    Visualize backtesting results with quarterly and cumulative returns.
    
    Args:
        avg_return: DataFrame with return metrics
        title_suffix: Suffix to add to plot titles
    """
    avg_return_copy = avg_return.copy()
    avg_return_copy['quarter'] = avg_return_copy['quarter'].astype(str)

    # Plot 1: Quarterly Returns
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.patch.set_facecolor('black')
    ax1.set_facecolor('black')

    ax1.plot(avg_return_copy['quarter'], avg_return_copy['nlp_return'], 
             label='NLP Strategy', color='#FF6B6B', linewidth=2.5, marker='o')
    ax1.plot(avg_return_copy['quarter'], avg_return_copy['HS300Index'], 
             label='HS300 Index', color='#4ECDC4', linewidth=2.5, marker='s')
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.7)
    ax1.set_title(f'Quarterly Returns {title_suffix}', color='white', fontsize=14)
    ax1.set_ylabel('Return (%)', color='white')
    
    legend = ax1.legend(facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")

    ax1.grid(True, alpha=0.3, color='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    plt.show()

    # Plot 2: Cumulative Returns
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig2.patch.set_facecolor('black')
    ax2.set_facecolor('black')

    ax2.plot(avg_return_copy['quarter'], avg_return_copy['nlp_cum_return'], 
             label='NLP Strategy (Cumulative)', color='#FF6B6B', linewidth=2.5, marker='o')
    ax2.plot(avg_return_copy['quarter'], avg_return_copy['HS300_cum_return'], 
             label='HS300 Index (Cumulative)', color='#4ECDC4', linewidth=2.5, marker='s')
    ax2.set_title(f'Cumulative Returns {title_suffix}', color='white', fontsize=14)
    ax2.set_ylabel('Cumulative Return', color='white')
    
    legend = ax2.legend(facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color("white")
    
    ax2.grid(True, alpha=0.3, color='white')
    plt.xticks(rotation=45, ha='right', color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    plt.show()


def calculate_performance_metrics(avg_return):
    """
    Calculate Sharpe ratio and Information ratio for the strategy.
    
    Args:
        avg_return: DataFrame with return data
    
    Returns:
        dict with performance metrics
    """
    # Sharpe Ratio for NLP Strategy
    mean_nlp = avg_return['nlp_return'].mean()
    std_nlp = avg_return['nlp_return'].std()
    sharpe_nlp = mean_nlp / std_nlp if std_nlp != 0 else np.nan

    # Sharpe Ratio for CSI300
    mean_csi = avg_return['HS300Index'].mean()
    std_csi = avg_return['HS300Index'].std()
    sharpe_csi = mean_csi / std_csi if std_csi != 0 else np.nan

    # Information Ratio: NLP vs CSI300
    excess_nlp = avg_return['nlp_return'] - avg_return['HS300Index']
    mean_excess_nlp = excess_nlp.mean()
    std_excess_nlp = excess_nlp.std()
    info_ratio_nlp = mean_excess_nlp / std_excess_nlp if std_excess_nlp != 0 else np.nan

    # Total returns
    final_nlp_return = avg_return['nlp_cum_return'].iloc[-1]
    final_csi_return = avg_return['HS300_cum_return'].iloc[-1]

    metrics = {
        'nlp_total_return': final_nlp_return,
        'csi300_total_return': final_csi_return,
        'nlp_sharpe_ratio': sharpe_nlp,
        'csi300_sharpe_ratio': sharpe_csi,
        'nlp_information_ratio': info_ratio_nlp,
        'excess_return': final_nlp_return - final_csi_return,
    }

    return metrics


def print_performance_report(metrics):
    """
    Print formatted performance report.
    
    Args:
        metrics: dict with performance metrics
    """
    print("\n" + "="*60)
    print("📊 BACKTESTING RESULTS")
    print("="*60)
    
    print("\n📈 Total Cumulative Returns:")
    print(f"  NLP Strategy:  {metrics['nlp_total_return']:>8.2%}")
    print(f"  HS300 Index:   {metrics['csi300_total_return']:>8.2%}")
    print(f"  Excess Return: {metrics['excess_return']:>8.2%}")
    
    print("\n📊 Risk-Adjusted Performance Metrics:")
    print(f"  NLP Sharpe Ratio:         {metrics['nlp_sharpe_ratio']:>8.4f}")
    print(f"  HS300 Sharpe Ratio:       {metrics['csi300_sharpe_ratio']:>8.4f}")
    print(f"  NLP Information Ratio:    {metrics['nlp_information_ratio']:>8.4f}")
    
    print("="*60 + "\n")


def run_backtest(weights_file, strategy_name="Strategy", visualize=True):
    """
    Run complete backtesting pipeline for a given weights file.
    
    Args:
        weights_file: Path to quarterly weights CSV file
        strategy_name: Name of the strategy for reporting
        visualize: Whether to display plots
    
    Returns:
        dict with backtest results
    """
    print(f"\n{'='*60}")
    print(f"RUNNING BACKTEST: {strategy_name}")
    print(f"{'='*60}")
    
    # Load or prepare quarterly returns
    try:
        q_data = pd.read_csv('quarter_return.csv')
        print("[*] Loaded existing quarterly returns")
    except FileNotFoundError:
        q_data = prepare_quarterly_returns()
    
    # Prepare data
    q_data['quarter'] = pd.PeriodIndex(q_data['quarter'], freq='Q')
    hs_data = q_data[['quarter', '沪深300']].copy()
    hs_data.rename(columns={'沪深300': 'HS300Index'}, inplace=True)
    q_data.drop('沪深300', axis=1, inplace=True)

    # Load weights
    print(f"[*] Loading weights from {weights_file}...")
    weights = pd.read_csv(weights_file)
    weights['quarter'] = pd.PeriodIndex(weights['quarter'].str.replace('_', ''), freq='Q')
    weights['stock'] = weights['stock'].astype(str).str.zfill(6)
    print(f"✓ Loaded {len(weights)} weight records")

    # Run backtest
    print("[*] Running backtest...")
    avg_return = backtest(q_data, weights)
    avg_return = avg_return.droplevel(1)
    avg_return.name = 'nlp_return'
    avg_return = avg_return.reset_index()

    avg_return = avg_return.merge(hs_data, on='quarter', how='outer')

    # Process returns (convert from percentage)
    avg_return['nlp_return'] = avg_return['nlp_return'] / 100
    avg_return['HS300Index'] = avg_return['HS300Index'] / 100

    # Calculate cumulative and monthly returns
    avg_return['nlp_cum_return'] = (1 + avg_return['nlp_return']).cumprod() - 1
    avg_return['HS300_cum_return'] = (1 + avg_return['HS300Index']).cumprod() - 1
    avg_return['nlp_monthly_return'] = (1 + avg_return['nlp_return'])**(1/3) - 1
    avg_return['HS300_monthly_return'] = (1 + avg_return['HS300Index'])**(1/3) - 1

    # Calculate metrics
    metrics = calculate_performance_metrics(avg_return)
    print_performance_report(metrics)

    # Visualize
    if visualize:
        print("[*] Generating visualizations...")
        visualize_returns(avg_return, title_suffix=f"({strategy_name})")

    return {
        'strategy_name': strategy_name,
        'returns': avg_return,
        'metrics': metrics,
        'weights_file': weights_file
    }


def validate_weights(weights_file):
    """
    Validate weights data for consistency and issues.
    
    Args:
        weights_file: Path to weights CSV file
    """
    print(f"\n[*] Validating weights: {weights_file}")
    df = pd.read_csv(weights_file)
    df['stock'] = df['stock'].astype(str).str.zfill(6)

    # Check for duplicates
    dup_rows = df[df.duplicated(subset=['quarter', 'stock'], keep=False)]
    print(f"✓ Duplicate (quarter, stock) entries: {len(dup_rows)}")

    # Check stock appearances across quarters
    stock_counts = df.groupby('stock')['quarter'].nunique()
    multi_quarters = stock_counts[stock_counts > 1]
    print(f"✓ Stocks appearing in multiple quarters: {len(multi_quarters)}")

    # Check quarter format
    print(f"✓ Quarter samples: {df['quarter'].unique()[:5]}")
    print(f"✓ Total records: {len(df)}")
