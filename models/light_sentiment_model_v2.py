import re

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except ImportError:
    stemmer = None


class LightSentimentAnalyzerV2:
    def __init__(self):
        self.version = "light-v2"

        # ============================
        # Lexicon dengan Bobot
        # ============================
        self.lex_pos = {
            "bagus": 1,
            "baik": 1,
            "mantap": 1,
            "luar biasa": 2,
            "recommended": 2,
            "puas": 1,
            "senang": 1,
            "keren": 1,
            "nyaman": 1,
            "cepat": 1,
        }

        self.lex_neg = {
            "buruk": 1,
            "jelek": 1,
            "parah": 1,
            "mengecewakan": 2,
            "lambat": 1,
            "susah": 1,
            "payah": 1,
            "benci": 2,
            "error": 1,
            "lemot": 1,
        }

        # ============================
        # Intensifier / Downtoners
        # ============================
        self.intensifiers = {
            "sangat": 1.5,
            "banget": 1.5,
            "sekali": 1.5,
            "amat": 1.3,
        }

        self.downtoners = {
            "cukup": 0.6,
            "agak": 0.6,
            "lumayan": 0.7,
        }

        # ============================
        # Negation words
        # ============================
        self.negations = {"tidak", "tak", "ga", "gak", "enggak", "nggak", "bukan"}

    # =====================================
    # Clean text
    # =====================================
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if stemmer:
            text = stemmer.stem(text)
        return text

    # =====================================
    # Main Prediction
    # =====================================
    def predict(self, text):
        cleaned = self.clean_text(text)
        words = cleaned.split()

        total_score = 0
        invert_next = False

        for i, w in enumerate(words):
            # ============================
            # Negation Handling
            # ============================
            if w in self.negations:
                invert_next = True
                continue

            # ============================
            # Cek intensifier/downtoner
            # ============================
            modifier = 1
            if i > 0:
                prev = words[i - 1]
                if prev in self.intensifiers:
                    modifier = self.intensifiers[prev]
                elif prev in self.downtoners:
                    modifier = self.downtoners[prev]

            # ============================
            # Cek kata di lexicon
            # ============================
            score = 0

            if w in self.lex_pos:
                score = self.lex_pos[w] * modifier
            elif w in self.lex_neg:
                score = -self.lex_neg[w] * modifier

            # Negation flips sentiment
            if invert_next:
                score = -score
                invert_next = False

            total_score += score

        # ============================
        # Mapping score → Label
        # ============================
        if total_score > 0.5:
            sentiment = "positif"
        elif total_score < -0.5:
            sentiment = "negatif"
        else:
            sentiment = "netral"

        return {
            "label": sentiment,
            "score": float(total_score),
            "cleaned_text": cleaned,
            "model": self.version
        }
