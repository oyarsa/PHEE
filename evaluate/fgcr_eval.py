from __future__ import annotations
import argparse

from dataclasses import dataclass
from pathlib import Path

from sklearn import metrics


@dataclass
class Metric:
    name: str
    precision: float
    recall: float
    f1: float
    em: float

    def __str__(self) -> str:
        return (
            f"{self.name}\n"
            f"Precision: {self.precision:.2%}\n"
            f"Recall:    {self.recall:.2%}\n"
            f"F1:        {self.f1:.2%}\n"
            f"EM:        {self.em:.2%}"
        )


def evaluate(gold: list[list[str]], predicted: list[list[str]], name: str) -> Metric:
    """Evaluate labels for F1 and Exact Match

    Args:
        gold (list[list[str]]): gold instances, each a lists of gold labels
        predicted (list[list[str]]):
           predicted instances, each a list of predicted labels

    Returns:
        Metric: set of metrics we track
    """
    exact_match = 0
    y_gold: list[str] = []
    y_predicted: list[str] = []

    for g, p in zip(gold, predicted):
        exact_match += all(x == y for x, y in zip(g, p))
        y_gold.extend(g)
        y_predicted.extend(p)

    p, r, f1, _ = metrics.precision_recall_fscore_support(
        y_gold, y_predicted, average="macro"
    )
    em = exact_match / len(gold)

    return Metric(
        name=name,
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        em=em,
    )


@dataclass
class Entry:
    token: str
    gold: str
    pred: str
    prob: float


def evaluate_ace(path: Path) -> None:
    sentences: list[list[Entry]] = []
    sentence: list[Entry] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                sentences.append(sentence)
                sentence = []
            else:
                token, gold, pred, prob = line.split()
                sentence.append(Entry(token, gold, pred, float(prob)))

    golds = [[e.gold for e in sent] for sent in sentences]
    preds = [[e.pred for e in sent] for sent in sentences]
    result = evaluate(golds, preds, "BERT - Test")
    print(result)


evals = {
    "ace": evaluate_ace,
}


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", help=f"Model to evaluate. One of {list(evals)}.")
    argparser.add_argument("path", help="Path to predictions file or folder.")

    args = argparser.parse_args()
    model = args.model.lower().strip()
    path = Path(args.path)

    if model not in evals:
        raise ValueError("Invalid model: " + args.model)

    print(evals[model](path))


if __name__ == "__main__":
    main()
