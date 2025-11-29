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
    # final predict: ensemble
    # -------------------------
    def predict(self, text):
        # rule-based score (range arbitrary)
        rule_s = self.rule_score(text)

        # onnx model probabilistic prediction
        proba = self.onnx_predict_proba(text)  # may be None
        if proba is None:
            # fallback: map rule_score -> label
            if rule_s > 0.5:
                return {"label": "positif", "score": rule_s, "model": self.version}
            elif rule_s < -0.5:
                return {"label": "negatif", "score": abs(rule_s), "model": self.version}
            else:
                return {"label": "netral", "score": 0.5, "model": self.version}

        # compute onnx_score as single float in [-1,1] (pos-neg)
        if "pos" in proba and "neg" in proba:
            onnx_score = float(proba["pos"] - proba["neg"])
            onnx_conf = float(proba.get("pos", 0.0))
        elif "pos" in proba and "neg" not in proba:
            onnx_score = float(proba.get("pos", 0.0))
            onnx_conf = onnx_score
        else:
            # fallback: use first/last
            vals = list(proba.values())
            onnx_score = float(vals[-1] - vals[0]) if len(vals) >= 2 else float(vals[0])
            onnx_conf = float(max(vals))

        # normalize rule score to roughly [-1,1] using tanh-like scaling
        rule_norm = np.tanh(rule_s / 3.0)  # adjust divisor if needed

        # ensemble: weighted average
        w = self.onnx_weight
        ensemble_score = w * onnx_score + (1 - w) * rule_norm

        # map ensemble_score to label
        if ensemble_score > 0.2:
            label = "positif"
            conf = min(1.0, w * onnx_conf + (1 - w) * max(0.5, rule_norm))
        elif ensemble_score < -0.2:
            label = "negatif"
            conf = min(1.0, w * (1 - onnx_conf) + (1 - w) * max(0.5, -rule_norm))
        else:
            label = "netral"
            conf = 0.5

        return {"label": label, "score": float(conf), "model": self.version, "ensemble_score": float(ensemble_score)}
