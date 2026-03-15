# import os
# import re
# import csv
# import nltk
# import pandas as pd
# from transformers import pipeline
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm

# # Download stopwords (needed on first run)
# nltk.download('stopwords')

# def preprocess_text(text):
#     # Normalize whitespace before analysis
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# def sentiment_analysis(text, sentiment_analyzer):
#     # Split long text into 500-char chunks and run sentiment model
#     text = preprocess_text(text)
#     results = []
#     try:
#         chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
#         for chunk in chunks:
#             if len(chunk.strip()) > 10:
#                 result = sentiment_analyzer(chunk[:500], truncation=True)[0]
#                 results.append(result)
#     except Exception as e:
#         print(f"Sentiment analysis error: {e}")
#     return results

# def read_text(file_url, sentiment_analyzer):
#     # Read one file then run sentiment analysis
#     with open(file_url, 'r', encoding='utf-8') as f:
#         english_text = f.read()
#     return sentiment_analysis(english_text, sentiment_analyzer)

# def get_all_files_url(root_dir):
#     # Collect all file paths under root_dir recursively
#     all_files = []
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.lower().endswith(".txt"):
#                 file_path = os.path.join(root, file)
#                 all_files.append(file_path)
#     return all_files

# def sentiment_score_combination(root_dir, sentiment_analyzer):
#     # Aggregate sentiment scores across all files for each stock (per day)
#     all_files_url = get_all_files_url(root_dir)
#     print("all_files_url of", root_dir, "has", len(all_files_url), "paths.")

#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     df = pd.read_csv(os.path.join(base_dir, "Eastmoney_report_pdf_download", "HS300.csv"), dtype=str)
#     code_list = list(df['股票代码'])
#     name_list = list(df['股票简称'])

#     pos_values = [0] * len(code_list)
#     neg_values = [0] * len(code_list)
#     confidence_sums = [0] * len(code_list)
#     number_list = [0] * len(code_list)

#     with ThreadPoolExecutor(max_workers=8) as executor:
#         futures = {executor.submit(read_text, file_url, sentiment_analyzer): file_url for file_url in all_files_url}
#         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
#             file_url = futures[future]
#             try:
#                 results = future.result()
#                 if not results:
#                     continue
#                 # 股票名是文件所在的父目录
#                 stock_name = os.path.basename(os.path.dirname(file_url))
#                 position = None
#                 for i, x in enumerate(name_list):
#                     if x == stock_name:
#                         position = i
#                         break
#                 if position is None:
#                     continue

#                 # Accumulate sentiment scores and confidence per stock
#                 for res in results:
#                     label = res.get('label')
#                     confidence = res.get('score')
#                     if confidence is None:
#                         continue
#                     if label == 'POSITIVE':
#                         pos_values[position] += confidence
#                     elif label == 'NEGATIVE':
#                         neg_values[position] += confidence
#                     confidence_sums[position] += confidence
#                     number_list[position] += 1
#             except Exception as e:
#                 print(f"Error processing {file_url}: {e}")

#     combine_list = []
#     for i in range(len(code_list)):
#         total = pos_values[i] + neg_values[i]
#         if number_list[i] == 0 or total == 0:
#             pos_score = 0
#             neg_score = 0
#             avg_conf = 0
#         else:
#             pos_score = pos_values[i] / total
#             neg_score = 1 - pos_score
#             avg_conf = confidence_sums[i] / number_list[i]

#         # Store normalized positive/negative scores and average confidence
#         combine_list.append((
#             code_list[i],
#             name_list[i],
#             pos_score,
#             neg_score,
#             avg_conf
#         ))

#     return combine_list


# if __name__ == "__main__":
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     path = os.path.join(base_dir, "reports_txt_by_day")

#     sentiment_analyzer = pipeline(
#         "sentiment-analysis",
#         model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#         revision="714eb0f",
#         device=0
#     )

#     os.makedirs("sentiment_scores_by_day", exist_ok=True)
#     os.chdir("sentiment_scores_by_day")

#     # 遍历每日文件夹
#     daily_dirs = os.listdir(path)

#     for daily_dir in daily_dirs:
#         print("Processing daily folder:", daily_dir)
#         combined_list = sentiment_score_combination(os.path.join(path, daily_dir), sentiment_analyzer)
#         print("combined list is written successfully!")

#         with open(daily_dir + '_data.csv', 'w', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             writer.writerow([
#                 'stock',
#                 'name',
#                 'positive_score',
#                 'negative_score',
#                 'avg_confidence'
#             ])
#             print("writing to csv now...")
#             writer.writerows(combined_list)
#             print("writing successfully")

# import os
# import csv
# import pandas as pd
# from collections import defaultdict

# def build_daily_sentiment(input_dir, output_dir, horizon=90):
#     """
#     Build daily sentiment scores:
#     每日情绪指数 = 过去 horizon 天内所有研报的均值
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # 存储每只股票的研报记录：stock -> list of (date, sentiment_score, name)
#     report_records = defaultdict(list)

#     for file in sorted(os.listdir(input_dir)):
#         if not file.endswith("_data.csv"):
#             continue

#         date_str = file.replace("_data.csv", "")
#         if not (date_str.isdigit() and len(date_str) == 8):
#             continue

#         date = pd.to_datetime(date_str, format="%Y%m%d")
#         df = pd.read_csv(os.path.join(input_dir, file))

#         # 更新研报记录
#         for _, row in df.iterrows():
#             stock = str(row['stock']).zfill(6)
#             name = row['name']
#             pos = row['positive_score']
#             neg = row['negative_score']
#             conf = row['avg_confidence']

#             sentiment_score = (pos - neg) * conf
#             report_records[stock].append((date, sentiment_score, name))

#         # 计算当日每只股票的有效情绪指数
#         sentiment_today = []
#         for stock, reports in report_records.items():
#             active_scores = [s for d, s, _ in reports if (date - d).days < horizon]
#             if active_scores:
#                 avg_sentiment = sum(active_scores) / len(active_scores)
#             else:
#                 avg_sentiment = 0.0
#             name = reports[-1][2] if reports else ""
#             sentiment_today.append([date.strftime("%Y%m%d"), stock, name, avg_sentiment])

#         # 写出当日情绪指数文件
#         output_file = os.path.join(output_dir, f"{date.strftime('%Y%m%d')}_sentiment.csv")
#         with open(output_file, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(['date', 'stock', 'name', 'sentiment'])
#             writer.writerows(sentiment_today)

#         print(f"✅ Sentiment scores for {date.strftime('%Y%m%d')} written to {output_file}")


# if __name__ == "__main__":
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     input_dir = os.path.join(base_dir, "sentiment_scores_by_day")
#     output_dir = os.path.join(base_dir, "daily_sentiment_scores")

#     build_daily_sentiment(input_dir, output_dir, horizon=90)
import os
import re
import csv
import nltk
import pandas as pd
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# 下载停用词（首次运行需要）
nltk.download('stopwords')

def preprocess_text(text):
    """清理文本，规范空格"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentiment_analysis(text, sentiment_analyzer):
    """对文本分块做情绪分析"""
    text = preprocess_text(text)
    results = []
    try:
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
        for chunk in chunks:
            if len(chunk.strip()) > 10:
                result = sentiment_analyzer(chunk[:500], truncation=True)[0]
                results.append(result)
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
    return results

def read_text(file_url, sentiment_analyzer):
    """读取一个文件并做情绪分析"""
    with open(file_url, 'r', encoding='utf-8') as f:
        english_text = f.read()
    return sentiment_analysis(english_text, sentiment_analyzer)

def get_all_files_url(root_dir):
    """收集目录下所有txt文件路径"""
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    return all_files

def sentiment_score_combination(root_dir, sentiment_analyzer):
    """
    针对某一天的所有研报，计算每只股票的情绪分数
    返回列表：[stock, name, sentiment_score]
    """
    all_files_url = get_all_files_url(root_dir)
    print("all_files_url of", root_dir, "has", len(all_files_url), "paths.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "Eastmoney_report_pdf_download", "HS300.csv"), dtype=str)
    code_list = list(df['股票代码'])
    name_list = list(df['股票简称'])

    sentiment_scores = defaultdict(list)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(read_text, file_url, sentiment_analyzer): file_url for file_url in all_files_url}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_url = futures[future]
            try:
                results = future.result()
                if not results:
                    continue
                stock_name = os.path.basename(os.path.dirname(file_url))
                position = None
                for i, x in enumerate(name_list):
                    if x == stock_name:
                        position = i
                        break
                if position is None:
                    continue

                stock_code = code_list[position]
                stock_name = name_list[position]

                # 累积情绪分数
                for res in results:
                    label = res.get('label')
                    confidence = res.get('score')
                    if confidence is None:
                        continue
                    score = confidence if label == 'POSITIVE' else -confidence
                    sentiment_scores[stock_code].append((score, stock_name))
            except Exception as e:
                print(f"Error processing {file_url}: {e}")

    combine_list = []
    for code in code_list:
        if sentiment_scores[code]:
            avg_sentiment = sum(s for s, _ in sentiment_scores[code]) / len(sentiment_scores[code])
            name = sentiment_scores[code][0][1]
        else:
            avg_sentiment = 0.0
            name = name_list[code_list.index(code)]
        combine_list.append((code, name, avg_sentiment))

    return combine_list

def build_daily_sentiment(path, output_dir, horizon=90):
    """
    构建每日情绪指数：过去 horizon 天内所有研报的均值
    """
    os.makedirs(output_dir, exist_ok=True)

    # 存储每只股票的研报记录：stock -> list of (date, sentiment_score, name)
    report_records = defaultdict(list)

    daily_dirs = sorted(os.listdir(path))

    for daily_dir in daily_dirs:
        print("Processing daily folder:", daily_dir)
        combined_list = sentiment_score_combination(os.path.join(path, daily_dir), sentiment_analyzer)

        date = pd.to_datetime(daily_dir, format="%Y%m%d")

        # 更新研报记录
        for code, name, score in combined_list:
            report_records[code].append((date, score, name))

        # 计算当日每只股票的有效情绪指数（过去 horizon 天均值）
        sentiment_today = []
        for stock, reports in report_records.items():
            active_scores = [s for d, s, _ in reports if (date - d).days < horizon]
            if active_scores:
                avg_sentiment = sum(active_scores) / len(active_scores)
            else:
                avg_sentiment = 0.0
            name = reports[-1][2] if reports else ""
            sentiment_today.append([date.strftime("%Y%m%d"), stock, name, avg_sentiment])

        # 写出当日情绪指数文件
        output_file = os.path.join(output_dir, f"{date.strftime('%Y%m%d')}_sentiment.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'stock', 'name', 'sentiment'])
            writer.writerows(sentiment_today)

        print(f"✅ Sentiment scores for {date.strftime('%Y%m%d')} written to {output_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "reports_txt_by_day")

    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f",
        device=0
    )

    output_dir = os.path.join(base_dir, "daily_sentiment_scores")
    build_daily_sentiment(path, output_dir, horizon=90)
