import re

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except ImportError:
    stemmer = None


class LightSentimentAnalyzerV3:
    def __init__(self):
        self.version = "light-v3"

        # ========== LEXICON DASAR ==========
        self.lex_pos = {
            "bagus": 1,
            "baik": 1,
            "mantap": 1,
            "puas": 1,
            "keren": 1,
            "nyaman": 1,
            "cepat": 1,
            "recommended": 2,
            "luarbiasa": 2,
        }

        self.lex_neg = {
            "buruk": 1,
            "jelek": 1,
            "parah": 1,
            "mengecewakan": 2,
            "lambat": 1,
            "susah": 1,
            "payah": 1,
            "error": 1,
            "lemot": 1,
        }

        # ========== N-GRAM LEXICON ==========
        self.bigram_pos = {
            "sangat bagus": 2,
            "sangat baik": 2,
            "sangat puas": 2,
            "luar biasa": 2,
        }

        self.bigram_neg = {
            "sangat jelek": 2,
            "sangat buruk": 2,
            "tidak bagus": -2,
            "tidak puas": -2,
            "kurang bagus": -1.5,
            "kurang memuaskan": -2,
        }

        # ========== INTENSIFIER / DOWNTONER ==========
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

        self.negations = {"tidak", "tak", "ga", "gak", "enggak", "nggak", "bukan"}

        # ========== DOMAIN BOOSTER ==========
        self.domain_words = {
            "layanan": 1.2,
            "service": 1.2,
            "pengiriman": 1.2,
            "server": 1.2,
            "harga": 1.2,
            "fitur": 1.2,
            "aplikasi": 1.2,
        }

        # ========== TF-IDF MANUAL BOOST ==========
        self.common_positive = {"bagus", "baik", "puas", "keren", "mantap"}
        self.common_negative = {"buruk", "jelek", "kecewa", "lambat", "lemot", "error"}

    # =============== CLEANING ===============
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if stemmer:
            text = stemmer.stem(text)
        return text

    # =============== MAIN LOGIC ===============
    def predict(self, text):
        cleaned = self.clean_text(text)
        words = cleaned.split()

        total_score = 0
        invert_next = False

        # ========== CEK BIGRAM ==========
        joined = " ".join(words)
        for phrase, val in self.bigram_pos.items():
            if phrase in joined:
                total_score += val

        for phrase, val in self.bigram_neg.items():
            if phrase in joined:
                total_score += val

        # ========== WORD-LEVEL ==========
        for i, w in enumerate(words):

            # Negation
            if w in self.negations:
                invert_next = True
                continue

            modifier = 1
            if i > 0:
                prev = words[i - 1]
                if prev in self.intensifiers:
                    modifier = self.intensifiers[prev]
                elif prev in self.downtoners:
                    modifier = self.downtoners[prev]

            score = 0
            if w in self.lex_pos:
                score = self.lex_pos[w] * modifier
            elif w in self.lex_neg:
                score = -self.lex_neg[w] * modifier

            if invert_next:
                score = -score
                invert_next = False

            total_score += score

            # Domain booster
            if w in self.domain_words:
                total_score *= self.domain_words[w]

        # ========== TF-IDF MANUAL BOOST ==========
        freq = {w: words.count(w) for w in set(words)}

        for w, count in freq.items():
            if w in self.common_positive:
                total_score += 0.1 * count
            elif w in self.common_negative:
                total_score -= 0.1 * count

        # ========== FINAL LABEL ==========
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
