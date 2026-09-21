"""EM/F1 no padrão SQuAD/HotpotQA (minúsculas, sem pontuação, sem artigos)."""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # regra do avaliador oficial do HotpotQA para yes/no
    if p in {"yes", "no", "noanswer"} and p != g:
        return 0.0
    if g in {"yes", "no", "noanswer"} and p != g:
        return 0.0
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def best_em_f1(pred: str, golds: list[str]) -> tuple[float, float]:
    if not golds:
        return 0.0, 0.0
    return (max(exact_match(pred, g) for g in golds), max(f1_score(pred, g) for g in golds))


def support_recall(doc_titles: list[str], gold_titles: list[str]) -> float:
    """SR(s): fração dos títulos de suporte presentes no estado (alvo de treino do DynaKRAG)."""
    from .utils import norm_title

    gold = {norm_title(t) for t in gold_titles}
    if not gold:
        return 0.0
    got = {norm_title(t) for t in doc_titles}
    return len(gold & got) / len(gold)
