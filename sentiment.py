import os
import re
import csv
import nltk
import pandas as pd
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Download stopwords (needed on first run)
nltk.download('stopwords')

def preprocess_text(text):
    # Normalize whitespace before analysis
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def sentiment_analysis(text, sentiment_analyzer):
    # Split long text into 500-char chunks and run sentiment model
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
    # Read one file then run sentiment analysis
    with open(file_url, 'r', encoding='utf-8') as f:
        english_text = f.read()
    return sentiment_analysis(english_text, sentiment_analyzer)

def get_all_files_url(root_dir):
    # Collect all file paths under root_dir recursively
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files

def output_dir_name(root_dir):
    # List immediate children under root_dir
    return os.listdir(root_dir)

def sentiment_score_combination(root_dir, sentiment_analyzer):
    # Aggregate sentiment scores across all files for each stock
    all_files_url = get_all_files_url(root_dir)
    print("all_files_url of", root_dir, "has", len(all_files_url), "paths.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "Eastmoney_report_pdf_download", "HS300.csv"), dtype=str)
    code_list = list(df['股票代码'])
    name_list = list(df['股票简称'])

    pos_values = [0] * len(code_list)
    neg_values = [0] * len(code_list)
    confidence_sums = [0] * len(code_list)
    number_list = [0] * len(code_list)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(read_text, file_url, sentiment_analyzer): file_url for file_url in all_files_url}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_url = futures[future]
            try:
                results = future.result()
                if not results:
                    continue
                stock_name = os.path.basename(os.path.dirname(file_url))
                position = 0
                for i, x in enumerate(name_list):
                    if x == stock_name:
                        position = i
                        break

                # Accumulate sentiment scores and confidence per stock
                for res in results:
                    label = res.get('label')
                    confidence = res.get('score')
                    if confidence is None:
                        continue
                    if label == 'POSITIVE':
                        pos_values[position] += confidence
                    elif label == 'NEGATIVE':
                        neg_values[position] += confidence
                    confidence_sums[position] += confidence
                    number_list[position] += 1
            except Exception as e:
                print(f"Error processing {file_url}: {e}")

    combine_list = []
    for i in range(len(code_list)):
        total = pos_values[i] + neg_values[i]
        if number_list[i] == 0 or total == 0:
            pos_score = 0
            neg_score = 0
            avg_conf = 0
        else:
            pos_score = pos_values[i] / total
            neg_score = 1 - pos_score
            avg_conf = confidence_sums[i] / number_list[i]

        # Store normalized positive/negative scores and average confidence
        combine_list.append((
            code_list[i],
            name_list[i],
            pos_score,
            neg_score,
            avg_conf
        ))

    return combine_list




if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "reports_txt_by_quarter_cleaned_en")

    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f",
        device=0
    )

    os.makedirs("sentiment_scores_by_quarter_cleaned_en", exist_ok=True)
    os.chdir("sentiment_scores_by_quarter_cleaned_en")

    output_dir_names = output_dir_name(path)

    for output_dir_name in output_dir_names:
        print("combined list is:", output_dir_name)
        combined_list = sentiment_score_combination(os.path.join(path, output_dir_name), sentiment_analyzer)
        print("combined list is written successfully!")

        with open(output_dir_name + '_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'stock',
                'name',
                'positive_score',
                'negative_score',
                'avg_confidence'
            ])
            print("writing to csv now...")
            writer.writerows(combined_list)
            print("writing successfully")
