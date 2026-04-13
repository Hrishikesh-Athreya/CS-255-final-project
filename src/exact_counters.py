from collections import Counter


def exact_distinct_count(stream: list[str]) -> int:
    return len(set(stream))


def exact_frequencies(stream: list[str]) -> dict[str, int]:
    return dict(Counter(stream))


def exact_heavy_hitters(stream: list[str], threshold: float) -> dict[str, int]:
    freqs = exact_frequencies(stream)
    cutoff = threshold * len(stream)
    return {item: count for item, count in freqs.items() if count >= cutoff}
