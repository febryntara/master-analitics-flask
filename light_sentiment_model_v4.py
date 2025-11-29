# sentiment_model_light_v4.py
import re
import os
import numpy as np

# try optional fast stemmer
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except Exception:
    stemmer = None

# try load tokenizer (required if ONNX used)
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# try onnxruntime (inference engine)
try:
    import onnxruntime as ort
except Exception:
    ort = None


class LightSentimentAnalyzerV4:
    """
    Hybrid analyzer:
    - Use ONNX model (if available) for sequence-classification inference (fast, CPU).
    - Combine (weighted average) ONNX score with rule-based (v3) score.
    - Fallback to pure rule-based v3 if ONNX not available.
    """

    def __init__(
        self,
        onnx_model_path="onnx/model.onnx",
        onnx_tokenizer_name="indolem/indobert-base-p1",  # replace with your chosen small model
        onnx_weight=0.7,   # weight given to ONNX model in final ensemble
    ):
        self.version = "light-v4"
        self.onnx_model_path = onnx_model_path
        self.onnx_tokenizer_name = onnx_tokenizer_name
        self.onnx_weight = float(onnx_weight)

        # ----- rule-based v3 (embedded, simplified) -----
        self.lex_pos = {"bagus": 1, "baik": 1, "mantap": 1, "puas": 1, "keren": 1}
        self.lex_neg = {"buruk": 1, "jelek": 1, "parah": 1, "kecewa": 2, "lemot": 1}
        self.bigram_pos = {"sangat bagus": 2, "luar biasa": 2}
        self.bigram_neg = {"tidak bagus": -2, "tidak puas": -2, "kurang memuaskan": -2}
        self.intensifiers = {"sangat": 1.5, "banget": 1.5}
        self.downtoners = {"cukup": 0.6, "agak": 0.6}
        self.negations = {"tidak", "tak", "ga", "gak", "bukan", "nggak"}
        self.domain_words = {"layanan": 1.2, "pengiriman": 1.2, "harga": 1.2}

        # ----- ONNX runtime init (lazy) -----
        self.ort_session = None
        self.tokenizer = None
        if ort is not None and AutoTokenizer is not None and os.path.exists(self.onnx_model_path):
            try:
                self.ort_session = ort.InferenceSession(self.onnx_model_path, providers=["CPUExecutionProvider"])
                self.tokenizer = AutoTokenizer.from_pretrained(self.onnx_tokenizer_name, use_fast=True)
            except Exception:
                # if any failure, keep ort_session None and fallback to rule-based
                self.ort_session = None
                self.tokenizer = None

    # -------------------------
    # cleaning & stemming
    # -------------------------
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if stemmer:
            try:
                text = stemmer.stem(text)
            except Exception:
                pass
        return text

    # -------------------------
    # rule-based v3 scoring
    # -------------------------
    def rule_score(self, text):
        cleaned = self.clean_text(text)
        words = cleaned.split()
        total_score = 0
        invert_next = False

        joined = " ".join(words)
        for phrase, val in self.bigram_pos.items():
            if phrase in joined:
                total_score += val
        for phrase, val in self.bigram_neg.items():
            if phrase in joined:
                total_score += val

        for i, w in enumerate(words):
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

            if w in self.domain_words:
                total_score *= self.domain_words[w]

        # simple normalization: keep score in reasonable range
        return float(total_score)

    # -------------------------
    # ONNX inference (returns prob distribution-like)
    # expects model outputs logits for [neg, neu, pos] or [label probs]
    # -------------------------
    def onnx_predict_proba(self, text, max_length=128):
        if self.ort_session is None or self.tokenizer is None:
            return None

        cleaned = self.clean_text(text)
        enc = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        # map tokenizer outputs to session input names
        ort_inputs = {}
        for name in self.ort_session.get_inputs():
            inp_name = name.name
            # common names: input_ids, attention_mask, token_type_ids
            if inp_name in enc:
                ort_inputs[inp_name] = enc[inp_name].astype(np.int64)
            else:
                # fallback: if token_type_ids required but not present, create zeros
                if inp_name == "token_type_ids":
                    ort_inputs[inp_name] = np.zeros(enc["input_ids"].shape, dtype=np.int64)
                else:
                    # unknown input name: try to skip (some ONNX exports don't require extra)
                    pass

        outputs = self.ort_session.run(None, ort_inputs)
        # outputs could be logits; convert to probabilities
        logits = outputs[0]
        if logits.ndim == 2:
            from scipy.special import softmax
            probs = softmax(logits, axis=1)[0]
            # returns dict with keys 'neg','neu','pos' if 3 classes; else map generically
            if probs.shape[0] == 3:
                return {"neg": float(probs[0]), "neu": float(probs[1]), "pos": float(probs[2])}
            else:
                # for binary classification, treat index 0 negative, 1 positive
                if probs.shape[0] == 2:
                    return {"neg": float(probs[0]), "pos": float(probs[1])}
                else:
                    # fallback: return raw vector
                    return {"probs": probs.tolist()}
        else:
            return None

        # -------------------------
    # final predict: ensemble (score 0–1)
    # -------------------------
    def predict(self, text):
        # rule-based score (arbitrary range)
        rule_s = self.rule_score(text)

        # onnx proba (dict or None)
        proba = self.onnx_predict_proba(text)

        # -----------------------------
        # 1) Jika ONNX tidak ada → fallback rule-based dengan 0–1 confidence
        # -----------------------------
        if proba is None:
            # konversi rule_s ke skala [-1,1]
            rule_norm = np.tanh(rule_s / 3.0)
            # konversi ke skala 0–1
            rule_conf = (rule_norm + 1) / 2

            if rule_norm > 0.2:
                return {"label": "positif", "score": float(rule_conf), "model": self.version}
            elif rule_norm < -0.2:
                return {"label": "negatif", "score": float(1 - rule_conf), "model": self.version}
            else:
                return {"label": "netral", "score": 0.5, "model": self.version}

        # -----------------------------
        # 2) ONNX tersedia → kita ambil probabilitas murni (0–1) dari model
        # -----------------------------
        P_neg = float(proba.get("neg", 0.0))
        P_pos = float(proba.get("pos", 0.0))
        P_neu = float(proba.get("neu", 0.0)) if "neu" in proba else None

        # compute onnx score
        if P_neu is not None:
            # model 3 kelas
            onnx_score = P_pos - P_neg
        else:
            # model 2 kelas
            onnx_score = P_pos - P_neg

        # normalize rule score [-1..1]
        rule_norm = np.tanh(rule_s / 3.0)

        # ensemble combine (still -1..1)
        w = self.onnx_weight
        ensemble_score = w * onnx_score + (1 - w) * rule_norm

        # -----------------------------
        # 3) tentukan label
        # -----------------------------
        if ensemble_score > 0.2:
            label = "positif"
            conf = P_pos  # confidence 0–1 murni
        elif ensemble_score < -0.2:
            label = "negatif"
            conf = P_neg
        else:
            label = "netral"
            # kalau model 2 kelas → netral = 1 - (pos + neg)
            if P_neu is not None:
                conf = P_neu
            else:
                conf = 1 - (P_pos + P_neg)
                if conf < 0:
                    conf = 0.5  # fallback aman

        # pastikan 0–1
        conf = float(max(0.0, min(1.0, conf)))

        return {
            "label": label,
            "score": conf,             # ALWAYS 0–1
            "model": self.version,
            "ensemble_score": float(ensemble_score)
        }

