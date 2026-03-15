import os
import csv
import pandas as pd
from collections import defaultdict

def build_weight_change_table(input_dir, output_file):
    """
    Build weight change table (daily version) with constraints:
    1. Weight change happens instantly on report day
    2. After report day, weight change = 0
    3. At most 5 adjustments per stock per month
    (取消了平均值重置逻辑)
    """
    all_data = []

    # 每月调整次数
    adjustment_count = defaultdict(int)

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith("_data.csv"):
            continue

        date_str = file.replace("_data.csv", "")
        if not (date_str.isdigit() and len(date_str) == 8):
            continue

        date = pd.to_datetime(date_str, format="%Y%m%d")
        df = pd.read_csv(os.path.join(input_dir, file))

        for _, row in df.iterrows():
            stock = str(row['stock']).zfill(6)
            name = row['name']
            pos = row['positive_score']
            neg = row['negative_score']
            conf = row['avg_confidence']

            ym_key = (stock, date.year, date.month)

            # 如果超过5次调整，则跳过
            if adjustment_count[ym_key] >= 5:
                continue

            # 研报当天瞬时调整
            weight_change = (pos - neg) * conf

            adjustment_count[ym_key] += 1

            all_data.append([
                date_str,
                stock,
                name,
                pos,
                neg,
                conf,
                weight_change
            ])

    # 写出结果
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date',
            'stock',
            'name',
            'positive_score',
            'negative_score',
            'avg_confidence',
            'weight_change'
        ])
        writer.writerows(all_data)

    print(f"✅ Weight change table written to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "sentiment_scores_by_day")
    output_file = os.path.join(base_dir, "weight_change_daily.csv")

    build_weight_change_table(input_dir, output_file)
