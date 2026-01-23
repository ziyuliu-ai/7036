import os
import csv
import pandas as pd

def build_weight_change_table(input_dir, output_file):
    """
    Build weight change table
    input_dir: Directory containing quarterly result files
    output_file: Output weight change table CSV
    """
    all_data = []

    # Iterate through quarterly files
    for file in os.listdir(input_dir):
        if file.endswith("_data.csv"):
            quarter = file.replace("_data.csv", "")
            df = pd.read_csv(os.path.join(input_dir, file))

            for _, row in df.iterrows():
                stock = row['stock']
                name = row['name']
                pos = row['positive_score']
                neg = row['negative_score']
                conf = row['avg_confidence']

                # Weight change calculation: (positive - negative) * confidence
                weight_change = (pos - neg) * conf

                all_data.append([
                    quarter,
                    stock,
                    name,
                    pos,
                    neg,
                    conf,
                    weight_change
                ])

    # Write results to file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'quarter',
            'stock',
            'name',
            'positive_score',
            'negative_score',
            'avg_confidence',
            'weight_change'
        ])
        writer.writerows(all_data)

    print(f"Weight change table has been written to {output_file}")


if __name__ == "__main__":
    # Assume quarterly data is in sentiment_scores_by_quarter_cleaned_en folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "sentiment_scores_by_quarter_cleaned_en")
    output_file = os.path.join(base_dir, "weight_change.csv")

    build_weight_change_table(input_dir, output_file)
