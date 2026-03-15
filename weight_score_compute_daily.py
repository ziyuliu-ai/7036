
# import os
# import csv
# import pandas as pd

# def build_daily_weights(input_dir, output_file, top_n=10, bottom_n=10):
#     """
#     Build daily long-short portfolio based on sentiment scores.
#     Rules:
#     1. If >=20 stocks have non-zero sentiment:
#        - Long top_n with +10% each
#        - Short bottom_n with -10% each
#     2. If <20 stocks have non-zero sentiment:
#        - Long positive sentiment stocks equally
#        - Short negative sentiment stocks equally
#        - Remaining zero-sentiment stocks split equally long/short to ensure net=0
#     """

#     results = []

#     for file in sorted(os.listdir(input_dir)):
#         if not file.endswith("_sentiment.csv"):
#             continue

#         df = pd.read_csv(os.path.join(input_dir, file))
#         date = pd.to_datetime(df['date'].iloc[0], format="%Y%m%d")

#         # 筛选非零情绪股票
#         nonzero_df = df[df['sentiment'] != 0].copy()
#         zero_df = df[df['sentiment'] == 0].copy()

#         if len(nonzero_df) >= 20:
#             # 排序
#             sorted_df = nonzero_df.sort_values(by="sentiment", ascending=False)

#             longs = sorted_df.head(top_n)
#             shorts = sorted_df.tail(bottom_n)

#             for _, row in longs.iterrows():
#                 results.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], 0.10])
#             for _, row in shorts.iterrows():
#                 results.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], -0.10])

#         else:
#             # 按正负情绪分组
#             pos_df = nonzero_df[nonzero_df['sentiment'] > 0]
#             neg_df = nonzero_df[nonzero_df['sentiment'] < 0]

#             n_pos = len(pos_df)
#             n_neg = len(neg_df)

#             net_value = 0.0

#             # 如果只有正股
#             if n_pos > 0 and n_neg == 0:
#                 w_pos = 0.1  # 每只股票固定 +0.1
#                 for _, row in pos_df.iterrows():
#                     results.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_pos])
#                 net_value = w_pos * n_pos

#             # 如果只有负股
#             elif n_neg > 0 and n_pos == 0:
#                 w_neg = -0.1  # 每只股票固定 -0.1
#                 for _, row in neg_df.iterrows():
#                     results.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_neg])
#                 net_value = w_neg * n_neg

#             # 如果正负都有
#             else:
#                 if n_pos > 0:
#                     w_pos = 0.5 / n_pos
#                     for _, row in pos_df.iterrows():
#                         results.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_pos])
#                     net_value += w_pos * n_pos
#                 if n_neg > 0:
#                     w_neg = -0.5 / n_neg
#                     for _, row in neg_df.iterrows():
#                         results.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_neg])
#                     net_value += w_neg * n_neg

#             # 用零股对冲净值
#             if len(zero_df) > 0 and abs(net_value) > 1e-12:
#                 w_zero = -net_value / len(zero_df)
#                 for _, row in zero_df.iterrows():
#                     results.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_zero])


#     # 写出结果
#     with open(output_file, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(['date', 'stock', 'name', 'sentiment', 'weight'])
#         writer.writerows(results)

#     print(f"✅ Daily long-short weights written to {output_file}")


# if __name__ == "__main__":
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     input_dir = os.path.join(base_dir, "daily_sentiment_scores")
#     output_file = os.path.join(base_dir, "daily_weights_sentiment.csv")

#     build_daily_weights(input_dir, output_file, top_n=10, bottom_n=10)

import os
import csv
import pandas as pd

def build_daily_weights(input_dir, output_file, top_n=10, bottom_n=10):
    results = []

    # 收集所有 sentiment 文件日期
    file_dates = {}
    for file in os.listdir(input_dir):
        if file.endswith("_sentiment.csv"):
            df = pd.read_csv(os.path.join(input_dir, file))
            date = pd.to_datetime(df['date'].iloc[0], format="%Y%m%d")
            file_dates[date] = df

    # 按日期范围逐天迭代
    all_dates = pd.date_range(min(file_dates.keys()), max(file_dates.keys()))
    prev_day_weights = None

    for date in all_dates:
        if date in file_dates:
            df = file_dates[date]

            nonzero_df = df[df['sentiment'] != 0].copy()
            zero_df = df[df['sentiment'] == 0].copy()

            day_weights = []

            if len(nonzero_df) >= 20:
                sorted_df = nonzero_df.sort_values(by="sentiment", ascending=False)
                longs = sorted_df.head(top_n)
                shorts = sorted_df.tail(bottom_n)

                for _, row in longs.iterrows():
                    day_weights.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], 0.10])
                for _, row in shorts.iterrows():
                    day_weights.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], -0.10])

            else:
                pos_df = nonzero_df[nonzero_df['sentiment'] > 0]
                neg_df = nonzero_df[nonzero_df['sentiment'] < 0]
                net_value = 0.0

                if len(pos_df) > 0:
                    w_pos = 0.5 / len(pos_df)
                    for _, row in pos_df.iterrows():
                        day_weights.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_pos])
                    net_value += w_pos * len(pos_df)

                if len(neg_df) > 0:
                    w_neg = -0.5 / len(neg_df)
                    for _, row in neg_df.iterrows():
                        day_weights.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_neg])
                    net_value += w_neg * len(neg_df)

                if len(zero_df) > 0 and abs(net_value) > 1e-12:
                    w_zero = -net_value / len(zero_df)
                    for _, row in zero_df.iterrows():
                        day_weights.append([date.strftime("%Y%m%d"), row['stock'], row['name'], row['sentiment'], w_zero])

            prev_day_weights = day_weights
            results.extend(day_weights)

        else:
            # 没有文件 → 复制上一日权重
            if prev_day_weights is not None:
                copied = [[date.strftime("%Y%m%d"), s, n, sent, w] for (_, s, n, sent, w) in prev_day_weights]
                results.extend(copied)

    # 写出结果
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'stock', 'name', 'sentiment', 'weight'])
        writer.writerows(results)

    print(f"✅ Daily long-short weights written to {output_file}")
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "daily_sentiment_scores")
    output_file = os.path.join(base_dir, "daily_weights_sentiment.csv")

    build_daily_weights(input_dir, output_file, top_n=10, bottom_n=10)
