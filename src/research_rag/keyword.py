"""Keyword search with BM25: chunks that contain the question's rarer words rank higher."""

import math
import re
from collections import Counter

import numpy as np

WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")  # keeps "gpt-4", "clip-s", "1.5"


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float32)
        self.avg_length = float(self.lengths.mean())
        df = Counter(word for d in self.docs for word in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for word in set(tokenize(query)):
            if word not in self.idf:
                continue
            tf = np.array([d[word] for d in self.docs], dtype=np.float32)
            out += self.idf[word] * tf * (self.k1 + 1) / (tf + norm)
        return out
