from sklearn import metrics
from typing import NamedTuple


class Metric(NamedTuple):
    f1: float
    em: float


def evaluate(gold: list[list[str]], predicted: list[list[str]]) -> Metric:
    """Evaluate labels for F1 and Exact Match

    Args:
        gold (list[list[str]]): gold instances, each a lists of gold labels
        predicted (list[list[str]]):
           predicted instances, each a list of predicted labels

    Returns:
        Metric: tuple of (F1, EM)
    """
    exact_match = 0
    y_gold: list[str] = []
    y_predicted: list[str] = []

    for g, p in zip(gold, predicted):
        exact_match += all(x == y for x, y in zip(g, p))
        y_gold.extend(g)
        y_predicted.extend(p)

    f1 = metrics.f1_score(y_gold, y_predicted, average="macro")
    em = exact_match / len(gold)
    return Metric(float(f1), em)
