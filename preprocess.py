import re
import time
from dics.normalization_dict_id import NORMALIZATION_MAP_ID
from models.light_sentiment_model_v4 import LightSentimentAnalyzerV4


# analyzer
analyzer = LightSentimentAnalyzerV4()

# stopword sederhana (bisa diperluas)
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","untuk","pada","itu","ini","karena",
    "atau","dengan","juga","ada","sudah","belum","bahwa","agar"
}

# ───────────────────────────────────────────────
# 1. BASIC CLEANING
# ───────────────────────────────────────────────
def basic_clean(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ───────────────────────────────────────────────
# 2. TOKENIZATION
# ───────────────────────────────────────────────
def tokenize(text):
    return text.split()


# ───────────────────────────────────────────────
# 3. NORMALIZATION (singkatan → kata baku)
# ───────────────────────────────────────────────
def normalize_tokens(tokens):
    return [NORMALIZATION_MAP_ID.get(t, t) for t in tokens]


# ───────────────────────────────────────────────
# 4. REDUCE REPEATED LETTERS
# ───────────────────────────────────────────────
def normalize_repeated(tokens):
    def fix(word):
        return re.sub(r'(.)\1{2,}', r'\1', word)
    return [fix(t) for t in tokens]


# ───────────────────────────────────────────────
# 5. STOPWORDS
# ───────────────────────────────────────────────
def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS_ID]


# ───────────────────────────────────────────────
# 6. FULL PIPELINE
# ───────────────────────────────────────────────
def run_preprocessing(raw_id, raw_text):
    start_time = time.time()
    
    # pre processing dimulai
    cleaned = basic_clean(raw_text)
    tokens = tokenize(cleaned)
    normalized = normalize_tokens(tokens)
    reduced = normalize_repeated(normalized)
    no_sw = remove_stopwords(reduced)
    final_text = " ".join(no_sw)

    # proses data ke model analizer 
    sentiment_result = analyzer.predict(final_text)
    sentiment = sentiment_result['label']
    confidence_score = sentiment_result['score']

    preprocessing_time_ms = int((time.time() - start_time) * 1000)

    # gas kirim output
    return {
        # data pelengkap, siapa tau perlu
        # "tokens": tokens,
        # "normalized": normalized,
        # "no_repeated": reduced,
        # "no_stopwords": no_sw,
        # "stemmed_tokens": stemmed,
        # data yang diterima database
        'raw_id' : raw_id,
        "cleaned_text": final_text,
        "sentiment": sentiment,
        "confidence_score": confidence_score,
        "preprocessing_time_ms": preprocessing_time_ms,
        "model_version": analyzer.version
    }
