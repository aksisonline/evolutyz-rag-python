"""segmentation.py
Utility functions for token-based text segmentation tuned for ColBERT best practices.

Environment variable suggestions (consumed by services):
PDF_CHUNK_TOKENS (default 180)
PDF_CHUNK_MAX_TOKENS (soft cap, default 300)
PDF_CHUNK_OVERLAP_TOKENS (default 0 or small 10-20)
PDF_CHUNK_HARD_MAX_TOKENS (absolute max, default 512)

CSV_CHUNK_TOKENS (default 180)
CSV_CHUNK_MAX_TOKENS (default 300)
CSV_CHUNK_OVERLAP_TOKENS (default 0)
CSV_CHUNK_HARD_MAX_TOKENS (default 512)

Rationale:
ColBERT retrieval typically performs best with 128–180 token chunks; increasing beyond 300 yields diminishing returns and may hurt precision.
Overlap is usually unnecessary (ColBERT scores per token) but a small overlap (<=20) can help preserve boundary context.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import math


def simple_tokenize(text: str) -> List[str]:  # lightweight tokenizer (whitespace + basic punctuation splitting)
    if not text:
        return []
    # Replace common punctuation with space-delimited versions to isolate tokens
    punct = "\n\t,.!?;:()[]{}<>\"'`|/\\"  # minimal set
    trans = str.maketrans({ch: " " for ch in punct})
    normalized = text.translate(trans)
    # Split on whitespace
    return [tok for tok in normalized.split() if tok]


def segment_text_by_tokens(
    text: str,
    target_tokens: int,
    soft_max_tokens: int,
    overlap_tokens: int,
    hard_max_tokens: int,
    safety_cap: int = 20000,
) -> List[str]:
    """Segment text into token windows following ColBERT chunk sizing rules.

    Args:
        text: Raw input string.
        target_tokens: Ideal chunk size (e.g., 128–180).
        soft_max_tokens: Soft upper bound; if a window slightly exceeds target but <= soft max, accept it.
        overlap_tokens: Tokens of backward overlap; often 0–20 for ColBERT.
        hard_max_tokens: Absolute max window length (truncate / force split if exceeded), usually 512.
        safety_cap: Max number of segments to prevent runaway splits.

    Returns:
        List of segmented text strings.
    """
    tokens = simple_tokenize(text)
    n = len(tokens)
    if n == 0:
        return []
    if n <= soft_max_tokens:
        return [text]

    segments: List[str] = []
    step = max(1, target_tokens - overlap_tokens) if target_tokens > overlap_tokens else target_tokens
    start = 0
    iterations = 0
    while start < n and iterations < safety_cap:
        iterations += 1
        end = start + target_tokens
        if end > n:
            end = n
        # Allow expansion up to soft max if next split would create a tiny fragment (< 30% target)
        remaining = n - end
        if 0 < remaining < int(0.3 * target_tokens) and end - start <= soft_max_tokens and end + remaining <= hard_max_tokens:
            end = n
        window_len = end - start
        if window_len > hard_max_tokens:
            end = start + hard_max_tokens
            window_len = hard_max_tokens
        segment_tokens = tokens[start:end]
        segments.append(" ".join(segment_tokens))
        if end >= n:
            break
        start = end - overlap_tokens if overlap_tokens > 0 else end
    return segments


def compute_dynamic_window(
    length_tokens: int,
    target_segment_count: int,
    min_tokens: int,
    max_tokens: int,
    hard_max_tokens: int,
) -> int:
    """Compute a dynamic window size so total segments ~= target_segment_count.

    window = clamp(ceil(length / target_segments), min_tokens, max_tokens) and never exceed hard_max.
    """
    if target_segment_count <= 0:
        target_segment_count = 1
    window = math.ceil(length_tokens / target_segment_count)
    window = max(min_tokens, min(max_tokens, window))
    window = min(window, hard_max_tokens)
    return window


def dynamic_segment_text(
    text: str,
    target_segment_count: int = 12,
    min_tokens: int = 120,
    max_tokens: int = 300,
    hard_max_tokens: int = 512,
    overlap_tokens: int = 0,
) -> List[str]:
    """Segment text with dynamic window size aiming for a given segment count.

    Logic:
      1. Tokenize once.
      2. If total tokens <= max_tokens -> return single segment.
      3. Compute window via compute_dynamic_window.
      4. Use segment_text_by_tokens with (target=window, soft_max=window*2 capped, overlap=overlap_tokens).
    """
    tokens = simple_tokenize(text)
    total = len(tokens)
    if total == 0:
        return []
    if total <= max_tokens:
        return [text]
    window = compute_dynamic_window(total, target_segment_count, min_tokens, max_tokens, hard_max_tokens)
    soft = min(hard_max_tokens, max(window, int(window * 1.5)))
    return segment_text_by_tokens(
        text=text,
        target_tokens=window,
        soft_max_tokens=soft,
        overlap_tokens=overlap_tokens,
        hard_max_tokens=hard_max_tokens,
    )


__all__ = [
    "simple_tokenize",
    "segment_text_by_tokens",
    "dynamic_segment_text",
    "compute_dynamic_window",
]
