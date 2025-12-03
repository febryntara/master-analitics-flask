# sentiment_model_light_v4_nostem.py
import re
import os
import numpy as np
from dics.lex_pos import LEX_POS
from dics.lex_neg import LEX_NEG
from dics.bigram_pos import BIGRAM_POS
from dics.bigram_neg import BIGRAM_NEG
from dics.intensifiers import INTENSIFIERS
from dics.downtoners import DOWNTONERS
from dics.negations import NEGATIONS
from dics.domain_words import DOMAIN_WORDS

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
    - Use ONNX model (if available).
    - Combine (weighted average) ONNX score with rule-based score.
    - Fallback to pure rule-based if ONNX not available.
    """

    def __init__(
        self,
        onnx_model_path="onnx/model.onnx",
        onnx_tokenizer_name="indolem/indobert-base-p1",
        onnx_weight=0.7,
    ):
        self.version = "light-v4-nostem"
        self.onnx_model_path = onnx_model_path
        self.onnx_tokenizer_name = onnx_tokenizer_name
        self.onnx_weight = float(onnx_weight)

        # ----- rule-based lexicons -----
        self.lex_pos = LEX_POS
        self.lex_neg = LEX_NEG
        self.bigram_pos = BIGRAM_POS
        self.bigram_neg = BIGRAM_NEG
        self.intensifiers = INTENSIFIERS
        self.downtoners = DOWNTONERS
        self.negations = NEGATIONS
        self.domain_words = DOMAIN_WORDS

        # ----- ONNX init -----
        self.ort_session = None
        self.tokenizer = None
        if ort is not None and AutoTokenizer is not None and os.path.exists(self.onnx_model_path):
            try:
                self.ort_session = ort.InferenceSession(
                    self.onnx_model_path,
                    providers=["CPUExecutionProvider"]
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.onnx_tokenizer_name,
                    use_fast=True
                )
            except Exception:
                self.ort_session = None
                self.tokenizer = None

    # -------------------------
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

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

        return float(total_score)

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

        ort_inputs = {}
        for name in self.ort_session.get_inputs():
            inp = name.name
            if inp in enc:
                ort_inputs[inp] = enc[inp].astype(np.int64)
            else:
                if inp == "token_type_ids":
                    ort_inputs[inp] = np.zeros(enc["input_ids"].shape, dtype=np.int64)

        outputs = self.ort_session.run(None, ort_inputs)
        logits = outputs[0]

        if logits.ndim == 2:
            from scipy.special import softmax
            probs = softmax(logits, axis=1)[0]

            if probs.shape[0] == 3:
                return {"neg": float(probs[0]), "neu": float(probs[1]), "pos": float(probs[2])}
            elif probs.shape[0] == 2:
                return {"neg": float(probs[0]), "pos": float(probs[1])}
            else:
                return {"probs": probs.tolist()}
        else:
            return None

    # -------------------------
    def predict(self, text):
        rule_s = self.rule_score(text)
        proba = self.onnx_predict_proba(text)

        if proba is None:
            rule_norm = np.tanh(rule_s / 3.0)
            rule_conf = (rule_norm + 1) / 2

            if rule_norm > 0.2:
                return {"label": "positif", "score": float(rule_conf), "model": self.version}
            elif rule_norm < -0.2:
                return {"label": "negatif", "score": float(1 - rule_conf), "model": self.version}
            else:
                return {"label": "netral", "score": 0.5, "model": self.version}

        P_neg = proba.get("neg", 0.0)
        P_pos = proba.get("pos", 0.0)
        P_neu = proba.get("neu", None)

        onnx_score = P_pos - P_neg

        rule_norm = np.tanh(rule_s / 3.0)
        ensemble = (self.onnx_weight * onnx_score + (1 - self.onnx_weight) * rule_norm)

        if ensemble > 0.2:
            return {"label": "positif", "score": float(ensemble), "model": self.version}
        elif ensemble < -0.2:
            return {"label": "negatif", "score": float(-ensemble), "model": self.version}
        else:
            return {"label": "netral", "score": float(abs(ensemble)), "model": self.version}
