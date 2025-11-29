from collections import Counter
from wordcloud import WordCloud
import nltk
nltk.download('punkt')

def word_frequency(text_list, top_n=30):
    words = []

    for text in text_list:
        tokens = nltk.word_tokenize(text.lower())
        words.extend(tokens)

    return Counter(words).most_common(top_n)


def generate_wordcloud(text_list, output_path="wordcloud.png"):
    combined = " ".join(text_list)

    wc = WordCloud(width=800, height=400, background_color="white")
    wc.generate(combined)
    wc.to_file(output_path)

    return output_path
