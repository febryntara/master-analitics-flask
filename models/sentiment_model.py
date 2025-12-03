from transformers import pipeline

class IndoBERTAnalyzer:
    def __init__(self):
        # Load model Hugging Face IndoBERT untuk sentiment analysis
        self.model_name = "taufiqdp/indonesian-sentiment"
        self.version = "taufiqdp/indonesian-sentiment"  # bisa juga versi semantik
        self.nlp = pipeline(
    "sentiment-analysis",
    model=self.model_name,
    tokenizer=self.model_name,
    framework="pt"
)

    def predict(self, text):
        """
        text: str, input mentah
        return: dict {label, score}
        """
        res = self.nlp(text)[0]  # hasil pipeline
        return {
            "label": res['label'].lower(),  # 'positive', 'neutral', 'negative'
            "score": float(res['score'])
        }
