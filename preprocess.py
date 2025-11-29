import time
# from sentiment_model import IndoBERTAnalyzer
from light_sentiment_model_v4 import LightSentimentAnalyzerV4


# analyzer = IndoBERTAnalyzer()
analyzer = LightSentimentAnalyzerV4()


def run_preprocessing(raw_id, raw_text):
    start_time = time.time()

    cleaned_text = raw_text.lower().strip()

    # Sentiment
    sentiment_result = analyzer.predict(cleaned_text)
    sentiment = sentiment_result['label']
    confidence_score = sentiment_result['score']

    preprocessing_time_ms = int((time.time() - start_time) * 1000)

    return {
        "raw_id": raw_id,
        "cleaned_text": cleaned_text,
        "sentiment": sentiment,
        "confidence_score": confidence_score,
        "preprocessing_time_ms": preprocessing_time_ms,
        "model_version": analyzer.version
    }
