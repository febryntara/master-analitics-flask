import re

class LightSentimentAnalyzer:
    def __init__(self):
        self.version = "light-v1"
    def __init__(self):
        # kamus kata positif
        self.positive = {
            "bagus", "baik", "mantap", "luar biasa", "keren", "top",
            "senang", "bahagia", "suka", "puas", "istimewa",
            "hebat", "mantul", "oke", "recommended", "bagus banget"
        }

        # kamus kata negatif
        self.negative = {
            "buruk", "jelek", "parah", "kecewa", "sedih", "marah",
            "kesal", "bangsat", "anjing", "tidak puas", "payah",
            "mengecewakan", "ampas", "hancur", "bodoh"
        }

    def clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict(self, text):
        cleaned = self.clean(text)
        words = cleaned.split()

        pos = 0
        neg = 0

        for w in words:
            if w in self.positive:
                pos += 1
            if w in self.negative:
                neg += 1

        # Skor confidence simple
        total = pos + neg
        if total == 0:
            return {"label": "neutral", "score": 0.50}

        score = max(pos, neg) / total

        if pos > neg:
            return {"label": "positive", "score": float(score)}
        elif neg > pos:
            return {"label": "negative", "score": float(score)}
        else:
            return {"label": "neutral", "score": float(score)}
