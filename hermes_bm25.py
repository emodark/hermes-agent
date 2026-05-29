#!/usr/bin/env python3
"""BM25 倒排索引 — 本地内存级全文搜索。

专为 Hermes Agent 会话搜索设计。基于 stockWeeklyAnalyzer/tools/bm25_index.py
的已验证实现，移除项目特定关联（同义词/词干提取）以保持轻量，保留核心：
- 倒排索引 + BM25 排名（k1=1.2, b=0.75）
- CJK 分词（单字索引）
- 前缀匹配
- JSON gzip 持久化
- 0 外部依赖（只用 stdlib）
"""

import gzip
import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── 分词器 ──────────────────────────────────────────────

# 英文 + 数字 token 正则
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x2E80, 0x2EFF),   # CJK Radicals Supplement
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Fullwidth Forms
    (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
]

_CJK_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "什么",
    "怎么", "如何", "为何", "因为", "所以", "但是", "如果", "虽然", "而且",
    "或者", "还是", "只是", "不过", "然后", "之后", "之前", "其中",
})

_EN_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "not", "no", "nor", "so", "if", "then", "else", "than",
    "too", "very", "just", "about", "also", "into", "over", "such", "only",
    "each", "other", "some", "any", "all", "both", "each", "few", "more",
    "most", "much", "same", "still", "well", "which", "who", "whom", "what",
    "when", "where", "why", "how",
})

_STOP_WORDS = _CJK_STOP_WORDS | _EN_STOP_WORDS


def _is_cjk(char: str) -> bool:
    """判断字符是否属于 CJK 范围。"""
    if len(char) != 1:
        return False
    code = ord(char)
    for start, end in _CJK_RANGES:
        if start <= code <= end:
            return True
    return False


def tokenize(text: str) -> List[str]:
    """分词：英文按空格/标点分，中文逐字分，过滤停用词和短词。"""
    if not text or not isinstance(text, str):
        return []

    text = text.lower()

    tokens: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            i += 1
            continue
        if _is_cjk(ch):
            tokens.append(ch)
            i += 1
            continue
        # 非 ASCII 可打印字符（emoji 等）→ 直接跳过
        if not (ch.isalnum() or ch == "_"):
            i += 1
            continue
        j = i
        while j < len(text) and (text[j].isalnum() or text[j] == "_") and not _is_cjk(text[j]):
            j += 1
        if j > i:
            token = text[i:j]
            if len(token) > 1 or token.isdigit():
                tokens.append(token)
        i = j

    result = []
    for token in tokens:
        if len(token) <= 1 and not token.isdigit() and not _is_cjk(token):
            continue
        if token in _STOP_WORDS:
            continue
        # CJK 单字不过滤长度；英文保留 >=2 字符
        if _is_cjk(token[0]) or len(token) >= 2:
            result.append(token)

    return result


def contains_cjk(text: str) -> bool:
    """判断文本是否包含 CJK 字符。"""
    if not text:
        return False
    return any(_is_cjk(ch) for ch in text)


# ── BM25 索引 ───────────────────────────────────────────

BM25Result = Dict[str, Any]


class BM25Index:
    """BM25 倒排索引。

    线程安全：所有公开方法加锁。
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # {id -> {id, content, content_preview, term_count, ...}}
        self._docs: Dict[str, Dict[str, Any]] = {}
        # {id -> {term -> count}}
        self._term_freqs: Dict[str, Dict[str, int]] = {}
        # {term -> Set[id]}
        self._inverted: Dict[str, Set[str]] = {}
        self._total_terms = 0
        self._sorted_terms: Optional[List[str]] = None
        self._dirty = False
        self._lock = __import__("threading").Lock()

    def doc_count(self) -> int:
        return len(self._docs)

    # ── 写入 ──

    def add(self, doc_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """索引一条文档。如果 doc_id 已存在则更新。"""
        if not content or not isinstance(content, str):
            return
        with self._lock:
            if doc_id in self._docs:
                self._remove_unlocked(doc_id)

            terms = tokenize(content)
            if not terms:
                return

            term_freq: Dict[str, int] = {}
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1

            self._docs[doc_id] = {
                "id": doc_id,
                "content": content,
                "content_preview": (content[:120] + "...") if len(content) > 120 else content,
                "term_count": len(terms),
                **(metadata or {}),
            }
            self._term_freqs[doc_id] = term_freq
            self._total_terms += len(terms)

            for term in term_freq:
                if term not in self._inverted:
                    self._inverted[term] = set()
                self._inverted[term].add(doc_id)

            self._sorted_terms = None
            self._dirty = True

    def remove(self, doc_id: str) -> None:
        """从索引中删除文档。"""
        with self._lock:
            self._remove_unlocked(doc_id)

    def _remove_unlocked(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        term_freq = self._term_freqs.get(doc_id, {})
        self._total_terms -= sum(term_freq.values())
        for term in term_freq:
            posting = self._inverted.get(term)
            if posting:
                posting.discard(doc_id)
                if not posting:
                    del self._inverted[term]
        self._term_freqs.pop(doc_id, None)
        self._docs.pop(doc_id, None)
        self._sorted_terms = None
        self._dirty = True

    def clear(self) -> None:
        """清空索引。"""
        with self._lock:
            self._docs.clear()
            self._term_freqs.clear()
            self._inverted.clear()
            self._total_terms = 0
            self._sorted_terms = None
            self._dirty = True

    # ── 搜索 ──

    def search(self, query: str, limit: int = 20) -> List[BM25Result]:
        """BM25 搜索。返回 [{id, score, content_preview, ...}]。"""
        if not query or not isinstance(query, str):
            return []
        if not self._docs:
            return []

        raw_terms = tokenize(query)
        if not raw_terms:
            return []

        with self._lock:
            if not self._inverted:
                return []

            N = len(self._docs)
            avg_doc_len = self._total_terms / N if N > 0 else 1

            # 去掉停用词后的搜索词
            q_terms = [t for t in raw_terms if t not in _STOP_WORDS]
            if not q_terms:
                q_terms = raw_terms

            scores: Dict[str, float] = {}
            for term in q_terms:
                posting = self._inverted.get(term)
                if not posting:
                    continue
                df = len(posting)
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                for doc_id in posting:
                    doc = self._docs[doc_id]
                    tf = self._term_freqs.get(doc_id, {}).get(term, 0)
                    doc_len = doc.get("term_count", 1)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * (doc_len / avg_doc_len)
                    )
                    scores[doc_id] = scores.get(doc_id, 0) + idf * (numerator / denominator)

                # 前缀匹配
                if self._sorted_terms is None:
                    self._sorted_terms = sorted(self._inverted.keys())
                start = self._lower_bound(self._sorted_terms, term)
                for si in range(start, len(self._sorted_terms)):
                    idx_term = self._sorted_terms[si]
                    if not idx_term.startswith(term):
                        break
                    if idx_term == term:
                        continue
                    posting2 = self._inverted[idx_term]
                    prefix_df = len(posting2)
                    prefix_idf = math.log((N - prefix_df + 0.5) / (prefix_df + 0.5) + 1) * 0.5
                    for doc_id in posting2:
                        tf = self._term_freqs.get(doc_id, {}).get(idx_term, 0)
                        doc_len = self._docs[doc_id].get("term_count", 1)
                        numerator = tf * (self.k1 + 1)
                        denominator = tf + self.k1 * (
                            1 - self.b + self.b * (doc_len / avg_doc_len)
                        )
                        scores[doc_id] = scores.get(doc_id, 0) + prefix_idf * (numerator / denominator)

            sorted_results = sorted(scores.items(), key=lambda x: -x[1])
            results = []
            for doc_id, score in sorted_results[:limit]:
                doc = self._docs.get(doc_id, {})
                results.append({
                    "id": doc_id,
                    "score": round(score, 4),
                    "content_preview": doc.get("content_preview", ""),
                    "content": doc.get("content", ""),
                    **{k: v for k, v in doc.items()
                       if k not in ("content", "content_preview", "term_count", "id")},
                })
            return results

    @staticmethod
    def _lower_bound(arr: List[str], target: str) -> int:
        """二分查找第一个 >= target 的位置。"""
        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ── 持久化 ──

    def save(self, path: str) -> None:
        """保存索引到 JSON 文件（gzip 压缩）。

        Thread-safe: snapshots all dicts under lock so json.dump() outside
        the lock never hits "dictionary changed size during iteration".
        """
        with self._lock:
            data = {
                "k1": self.k1,
                "b": self.b,
                "docs": dict(self._docs),
                "term_freqs": dict(self._term_freqs),
                "inverted": {k: list(v) for k, v in self._inverted.items()},
                "total_terms": self._total_terms,
            }
            self._dirty = False

        path = str(path)
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
            logger.debug("BM25Index saved to %s (%d docs)", path, len(data["docs"]))
        except Exception as exc:
            logger.error("BM25Index save failed: %s", exc)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """从文件加载索引。"""
        path = str(path)
        try:
            if path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.debug("BM25Index load skipped: %s", e)
            return cls()

        idx = cls(k1=data.get("k1", 1.2), b=data.get("b", 0.75))
        idx._docs = data.get("docs", {})
        idx._term_freqs = data.get("term_freqs", {})
        idx._inverted = {k: set(v) for k, v in data.get("inverted", {}).items()}
        idx._total_terms = data.get("total_terms", 0)
        idx._dirty = False
        logger.debug("BM25Index loaded from %s (%d docs)", path, len(idx._docs))
        return idx

    @property
    def is_dirty(self) -> bool:
        with self._lock:
            return self._dirty


# ── 工具函数 ──

def _fts5_to_bm25_query(query: str) -> str:
    """将 FTS5 查询语法转换为纯 BM25 搜索文本。

    主要转换：
    - 移除布尔运算符 AND/OR/NOT（BM25 隐式 AND）
    - 展开 quoted phrase 的引号
    - 保留通配符（BM25 自动前缀匹配）
    """
    if not query:
        return ""

    # 移除布尔运算符（BM25 不考虑它们）
    parts = re.split(r'\s+', query.strip())
    cleaned = []
    for p in parts:
        upper = p.upper()
        if upper in ("AND", "OR", "NOT"):
            continue
        # 移除引号
        p = p.strip('"\'')
        if p:
            cleaned.append(p)
    return " ".join(cleaned)
