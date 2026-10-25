"""Classical feature extractors: whitespace-token TF-IDF (+SVD) and Tree-Sitter AST features.

Follows paper Sec. 5.1:
 * TF-IDF: whitespace tokens, no lowercasing/stop-words/normalisation, 1-3 grams,
   5,000 n-gram features -> truncated SVD to 500 dims.
 * AST: depth, node-type counts, layout/style indicators (empty-line density), size proxies.
   Dimensionality is data-driven (distinct attributes seen in training).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------- language guess
_LANG_HINTS = [
    ("php", re.compile(r"<\?php|\$\w+\s*=")),
    ("rust", re.compile(r"\bfn\s+\w+\s*\(|\blet\s+mut\b|\bimpl\b")),
    ("go", re.compile(r"\bpackage\s+\w+|\bfunc\s+\w*\s*\(|:=")),
    ("csharp", re.compile(r"\busing\s+System|\bnamespace\s+\w+|\bpublic\s+static\s+void\s+Main")),
    ("java", re.compile(r"\bpublic\s+class\b|\bimport\s+java\.|System\.out\.print")),
    ("cpp", re.compile(r"#include\s*<[a-z_]+>\s*\n.*using\s+namespace|std::|\bcout\b", re.S)),
    ("c", re.compile(r"#include\s*<std\w+\.h>|\bprintf\s*\(")),
    ("javascript", re.compile(r"\bfunction\s*\w*\s*\(|=>|\bconsole\.log\b|\bconst\s+\w+\s*=")),
    ("python", re.compile(r"\bdef\s+\w+\s*\(|^\s*import\s+\w+|^\s*from\s+\w+\s+import", re.M)),
]


def guess_language(code: str) -> str:
    for name, pat in _LANG_HINTS:
        if pat.search(code):
            return name
    return "python"


_parsers = {}


def _get_parser(lang: str):
    if lang in _parsers:
        return _parsers[lang]
    parser = None
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        parser = get_parser({"csharp": "c_sharp"}.get(lang, lang))
    except Exception:
        parser = None
    _parsers[lang] = parser
    return parser


def _walk_ast(root):
    """Iterative walk: returns (type counter, max depth, n nodes)."""
    counts, max_depth, n = Counter(), 0, 0
    stack = [(root, 1)]
    while stack:
        node, d = stack.pop()
        n += 1
        max_depth = max(max_depth, d)
        counts[node.type] += 1
        for ch in node.children:
            stack.append((ch, d + 1))
    return counts, max_depth, n


def ast_features(code: str, language: Optional[str] = None) -> dict:
    lines = code.split("\n")
    lens = [len(l) for l in lines]
    feats = {
        "n_lines": len(lines),
        "n_chars": len(code),
        "max_line_len": max(lens) if lens else 0,
        "avg_line_len": float(np.mean(lens)) if lens else 0.0,
        "empty_line_density": sum(1 for l in lines if not l.strip()) / max(len(lines), 1),
        "alnum_frac": sum(c.isalnum() for c in code) / max(len(code), 1),
        "indent_tab_frac": sum(l.startswith("\t") for l in lines) / max(len(lines), 1),
        "comment_density": sum(l.lstrip().startswith(("#", "//", "/*", "*")) for l in lines)
                           / max(len(lines), 1),
    }
    lang = language or guess_language(code)
    parser = _get_parser(lang)
    if parser is not None:
        try:
            tree = parser.parse(code.encode("utf-8", errors="ignore"))
            counts, depth, n = _walk_ast(tree.root_node)
            feats["ast_depth"] = depth
            feats["ast_nodes"] = n
            for k, v in counts.items():
                feats[f"node::{k}"] = v / max(n, 1)  # normalised so lengths do not dominate
            for kw, pat in (("if", "if"), ("loop", "for|while"), ("func", "function|method")):
                feats[f"n_{kw}"] = sum(v for k, v in counts.items() if re.search(pat, k))
        except Exception:
            pass
    return feats


class AstFeaturizer:
    def __init__(self):
        self.vec = DictVectorizer(sparse=False)

    def fit_transform(self, codes: Iterable[str]) -> np.ndarray:
        return self.vec.fit_transform([ast_features(c) for c in codes])

    def transform(self, codes: Iterable[str]) -> np.ndarray:
        return self.vec.transform([ast_features(c) for c in codes])


class TfidfSvdFeaturizer:
    def __init__(self, max_features: int = 5000, svd_dim: int = 500, seed: int = 0):
        self.tfidf = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=False, stop_words=None,
            ngram_range=(1, 3), max_features=max_features, sublinear_tf=False)
        self.svd_dim = svd_dim
        self.seed = seed
        self.svd: Optional[TruncatedSVD] = None

    def fit_transform(self, codes: List[str]) -> np.ndarray:
        X = self.tfidf.fit_transform(codes)
        k = max(1, min(self.svd_dim, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        return self.svd.fit_transform(X)

    def transform(self, codes: List[str]) -> np.ndarray:
        return self.svd.transform(self.tfidf.transform(codes))


class CombinedFeaturizer:
    """TF-IDF(+SVD) and/or AST features. mode in {"tfidf", "ast", "both"}."""

    def __init__(self, mode: str = "tfidf", **tfidf_kw):
        assert mode in ("tfidf", "ast", "both")
        self.mode = mode
        self.tf = TfidfSvdFeaturizer(**tfidf_kw) if mode in ("tfidf", "both") else None
        self.ast = AstFeaturizer() if mode in ("ast", "both") else None

    def _stack(self, parts):
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def fit_transform(self, codes):
        parts = []
        if self.tf: parts.append(self.tf.fit_transform(codes))
        if self.ast: parts.append(self.ast.fit_transform(codes))
        return self._stack(parts)

    def transform(self, codes):
        parts = []
        if self.tf: parts.append(self.tf.transform(codes))
        if self.ast: parts.append(self.ast.transform(codes))
        return self._stack(parts)
