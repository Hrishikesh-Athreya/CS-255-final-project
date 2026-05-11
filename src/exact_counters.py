"""Exact (non-sketch) baselines for distinct counts, frequencies, and heavy hitters.

These functions materialize full sets or frequency tables and serve as ground truth
when comparing streaming sketches (FM, HLL, CMS).
"""

from collections import Counter


def exact_distinct_count(stream: list[str]) -> int:
    """Return the number of distinct elements in ``stream`` (cardinality).

    Args:
        stream: Token sequence; order is ignored; duplicates collapse to one id.

    Returns:
        Cardinality |{x : x appears in stream}|.
    """
    return len(set(stream))


def exact_frequencies(stream: list[str]) -> dict[str, int]:
    """Return exact occurrence counts for every token in ``stream``.

    Args:
        stream: Token sequence.

    Returns:
        Mapping from token to its count (sum of counts equals len(stream)).
    """
    return dict(Counter(stream))


def exact_heavy_hitters(stream: list[str], threshold: float) -> dict[str, int]:
    """Return tokens whose frequency is at least ``threshold`` × stream length.

    Args:
        stream: Token sequence.
        threshold: Fraction in (0, 1]; cutoff count is ``threshold * len(stream)``.

    Returns:
        Subset of exact frequencies for items meeting the cutoff.
    """
    freqs = exact_frequencies(stream)
    cutoff = threshold * len(stream)
    return {item: count for item, count in freqs.items() if count >= cutoff}
