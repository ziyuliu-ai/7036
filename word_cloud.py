import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')

# 1. Build stopword set
standard_stopwords = set(stopwords.words('english'))
custom_stopwords = {
    'company', 'companies', 'market', 'production', 'increase', 'profit', 'sales',
    'industry', 'expected', 'research', 'operations', 'products', 'investment', 'million', 'billion', 'yuan',
    'price', 'cost', 'business', 'data', 'report', 'year', 'quarter', 'first', 'half',
    'number', 'total', 'amount', 'per', 'cent', 'percent'
}
all_stopwords = standard_stopwords.union(custom_stopwords)

# 2. Filter English words (keep pure English tokens not in stopwords)
def filter_english_and_stopwords(words, stopwords):
    return [w for w in words if re.fullmatch(r'[a-zA-Z]+', w) and w not in stopwords]

# 3. Read data
file_path = "train.csv"
df = pd.read_csv(file_path)
df.rename(columns={df.columns[0]: 'text', df.columns[1]: 'label'}, inplace=True)

# 4. Clean text
df['text_clean'] = df['text'].astype(str).str.lower()

# 5. Group and count word frequency (after cleaning)
words_label4 = ' '.join(df[df['label'] == 4]['text_clean']).split()
words_label0 = ' '.join(df[df['label'] == 0]['text_clean']).split()

words_label4 = filter_english_and_stopwords(words_label4, all_stopwords)
words_label0 = filter_english_and_stopwords(words_label0, all_stopwords)

freq4 = Counter(words_label4)
freq0 = Counter(words_label0)

# 6. Compute frequency differences
all_words = set(freq4.keys()).union(set(freq0.keys()))
diff_freq_label4 = {w: freq4[w] - freq0[w] for w in all_words if freq4[w] > freq0[w]}
diff_freq_label0 = {w: freq0[w] - freq4[w] for w in all_words if freq0[w] > freq4[w]}

# 7. Generate word cloud helper
def plot_wordcloud_from_diff(word_freq, filename, title):
    if not word_freq:
        print(f"No differential words; skip {filename}")
        return
    
    wc = WordCloud(
        width=800,
        height=800,
        background_color='white'
    ).generate_from_frequencies(word_freq)
    
    wc.to_file(filename)

    # Also display the word cloud
    plt.figure(figsize=(6, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()
    
    print(f"{title} word cloud saved to {filename}")

# 8. Execute
def main():
    plot_wordcloud_from_diff(diff_freq_label4, "diff_wordcloud_label4.png", "Label4 top words")
    plot_wordcloud_from_diff(diff_freq_label0, "diff_wordcloud_label0.png", "Label0 top words")

if __name__ == "__main__":
    main()