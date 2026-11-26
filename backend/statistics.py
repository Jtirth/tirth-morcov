"""Small, inspectable helpers for corpus summary statistics."""
from collections import Counter


def frequency_distribution(tokens, limit=12):
    counts = Counter(tokens)
    total = len(tokens)
    return [{"word": word, "count": count, "relative_frequency": count / total if total else 0}
            for word, count in counts.most_common(limit)]


def probability_sum(predictions):
    return sum(item["probability"] for item in predictions)
