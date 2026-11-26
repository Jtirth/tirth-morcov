"""Frequency-based first- and second-order Markov text prediction."""
from collections import Counter, defaultdict
import re

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


def tokenize(text):
    """Normalize text to lowercase word tokens; punctuation separates words."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string.")
    return [match.group(0).lower().replace("’", "'") for match in TOKEN_RE.finditer(text)]


class MarkovModel:
    def __init__(self, text):
        self.tokens = tokenize(text)
        if not self.tokens:
            raise ValueError("Training text must contain at least one word.")
        self.unigrams = Counter(self.tokens)
        self.first = defaultdict(Counter)
        self.second = defaultdict(Counter)
        for index in range(len(self.tokens) - 1):
            self.first[self.tokens[index]][self.tokens[index + 1]] += 1
        for index in range(len(self.tokens) - 2):
            self.second[(self.tokens[index], self.tokens[index + 1])][self.tokens[index + 2]] += 1

    def predict(self, context, order=2, limit=5):
        if order not in (1, 2):
            raise ValueError("Order must be 1 or 2.")
        words = tokenize(context)
        if not words:
            return {"context": [], "order": order, "observations": 0, "predictions": []}
        used = words[-order:]
        counts = self.first.get(used[-1], Counter()) if order == 1 else self.second.get(tuple(used), Counter())
        total = sum(counts.values())
        all_predictions = [{"word": word, "count": count, "probability": count / total, "percent": 100 * count / total}
                           for word, count in counts.most_common()] if total else []
        return {"context": used, "order": order, "observations": total,
                "predictions": all_predictions[:max(1, min(int(limit), 20))],
                "distribution": all_predictions}

    def stats(self):
        first_rows = [{"context": context, "next": word, "count": count, "probability": count / sum(counts.values())}
                      for context, counts in self.first.items() for word, count in counts.most_common()]
        second_rows = [{"context": " ".join(context), "next": word, "count": count, "probability": count / sum(counts.values())}
                       for context, counts in self.second.items() for word, count in counts.most_common()]
        return {"total_words": len(self.tokens), "unique_words": len(self.unigrams),
                "distinct_first_contexts": len(self.first), "distinct_second_contexts": len(self.second),
                "top_words": [{"word": w, "count": c} for w, c in self.unigrams.most_common(10)],
                "first_transitions": first_rows[:80], "second_transitions": second_rows[:80],
                "first_transition_count": sum(map(sum, (c.values() for c in self.first.values()))),
                "second_transition_count": sum(map(sum, (c.values() for c in self.second.values())))}


def compare_models(training_text, test_text):
    """Evaluate each held-out token when its context occurs in training."""
    model = MarkovModel(training_text)
    test_tokens = tokenize(test_text)
    result = {}
    for order in (1, 2):
        total = covered = top1 = top3 = 0
        for index in range(order, len(test_tokens)):
            context, actual = test_tokens[index-order:index], test_tokens[index]
            total += 1
            distribution = model.predict(" ".join(context), order=order, limit=3)
            if distribution["observations"]:
                covered += 1
                guesses = [item["word"] for item in distribution["predictions"]]
                top1 += bool(guesses and guesses[0] == actual)
                top3 += actual in guesses
        result[str(order)] = {"order": order, "test_cases": total, "covered": covered,
                              "coverage": covered / total if total else 0,
                              "top1": top1 / covered if covered else 0,
                              "top3": top3 / covered if covered else 0,
                              "top1_hits": top1, "top3_hits": top3}
    return result
