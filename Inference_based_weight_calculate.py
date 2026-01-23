import os
import csv
import pandas as pd

def build_weight_change_table_regression(input_dir, output_file):
    """
    Build weight change table for regression version
    input_dir: folder containing quarterly results
    output_file: output weight change table CSV
    """
    all_data = []

    # Iterate through quarterly files
    for file in os.listdir(input_dir):
        if file.endswith(".csv") and file.startswith("scores_"):
            quarter = file.replace("scores_", "").replace(".csv", "").replace("Q", "_Q")
            df = pd.read_csv(os.path.join(input_dir, file))
            # Pad stock column with leading zeros to 6 digits
            # df['stock'] = df['stock'].astype(str).str.zfill(6)
            for _, row in df.iterrows():
                stock = row['stock']
                name = row['name']
                score_mean = row['score_mean']
                score_std = row['score_std']
                n_reports = row['n_reports']

                # Define confidence metric: more reports and lower volatility → higher confidence
                if pd.isna(score_std) or score_std == 0:
                    confidence = 1.0
                else:
                    confidence = 1 / (1 + score_std)

                # Weight change calculation formula (adjustable)
                weight_change = score_mean * confidence

                all_data.append([
                    quarter,
                    stock,
                    name,
                    score_mean,
                    score_std,
                    n_reports,
                    confidence,
                    weight_change
                ])

    # Write to summary table
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'quarter',
            'stock',
            'name',
            'score_mean',
            'score_std',
            'n_reports',
            'confidence',
            'weight_change'
        ])
        writer.writerows(all_data)

    print(f"✅ Regression version weight change table has been written to {output_file}")


if __name__ == "__main__":
    # Assume quarterly data is in the inference_outputs_regression folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "inference_outputs_regression")
    output_file = os.path.join(base_dir, "weight_change_inference.csv")

    build_weight_change_table_regression(input_dir, output_file)
