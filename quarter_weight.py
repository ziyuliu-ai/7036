import os
import csv
import pandas as pd

def build_quarterly_weights(input_file, output_file):
    """
    Update stock weights based on quarterly weight changes
    input_file: CSV containing weight_change for all quarters
    output_file: Output quarterly weight table
    """
    df = pd.read_csv(input_file)

    # Get all stocks
    stocks = df['stock'].unique()
    N = len(stocks)

    # Initialize weights uniformly
    weights = {stock: 1.0 / N for stock in stocks}

    results = []

    # Process in quarterly order
    for quarter in sorted(df['quarter'].unique()):
        quarter_df = df[df['quarter'] == quarter]

        # Update weights
        updated_weights = {}
        for _, row in quarter_df.iterrows():
            stock = row['stock']
            name = row['name']
            wc = row['weight_change']

            # Update formula
            updated_weights[stock] = weights[stock] * (1 + wc)

        # Normalize
        total = sum(updated_weights.values())
        for _, row in quarter_df.iterrows():
            stock = row['stock']
            name = row['name']
            wc = row['weight_change']
            new_weight = updated_weights[stock] / total

            results.append([quarter, stock, name, wc, new_weight])

        # Update to next quarter
        weights = {stock: updated_weights[stock] / total for stock in updated_weights}

    # Write results
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['quarter', 'stock', 'name', 'weight_change', 'weight'])
        writer.writerows(results)

    print(f"Quarterly weight table written to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "weight_change.csv")   # File generated in previous step
    output_file = os.path.join(base_dir, "quarterly_weights_sentiment.csv")

    input_file_1 = os.path.join(base_dir, "weight_change_inference.csv")   # File generated in previous step
    output_file_1 = os.path.join(base_dir, "quarterly_weights_inference.csv")

    build_quarterly_weights(input_file, output_file)
    build_quarterly_weights(input_file_1, output_file_1)
