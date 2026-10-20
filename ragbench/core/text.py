"""Texto: normalização, EM/F1 (padrão SQuAD/HotpotQA), extração de resposta e formatação de passagens."""
from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)
_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or is was are were be been being
    who what when where which why how whose whom did does do done that this these those
    as it its his her their he she they them there than then also into about over under
    both either same more most less least first last one two many much name named""".split()
)


# ----------------------------------------------------------------------------- métricas
def normalize_answer(s: str) -> str:
    """Normalização oficial do SQuAD/HotpotQA/FlashRAG: minúsculas, sem pontuação, sem artigos."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    # regra do avaliador oficial do HotpotQA para yes/no/noanswer
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
    """Máximo sobre as respostas de referência (como no artigo: 'the maximum score over references')."""
    golds = [g for g in (golds or []) if g is not None]
    if not golds:
        return 0.0, 0.0
    return max(exact_match(pred, g) for g in golds), max(f1_score(pred, g) for g in golds)


def contains_answer(pred: str, golds: list[str]) -> float:
    """Métrica 'acc' do FlashRAG: alguma resposta de referência aparece dentro da predição."""
    p = normalize_answer(pred)
    return float(any(normalize_answer(g) and normalize_answer(g) in p for g in golds or []))


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu_score(pred: str, gold: str, max_n: int = 4) -> float:
    """BLEU de uma resposta (Papineni et al., 2002), sobre os tokens normalizados como no EM.

    Segue o ``sentence_bleu`` do sacreBLEU com ``smooth_method="add-k"`` (k = 1, Lin e Och, 2004) e
    ``use_effective_order=True``: +1 nas contagens das ordens n ≥ 2, parada na primeira ordem sem n-gramas e
    média geométrica das precisões até essa ordem. Sem nenhum unigrama em comum, o BLEU é 0. Penalidade de
    brevidade: exp(1 − r/c) quando a predição é mais curta que o gabarito. Conferido contra o sacreBLEU 2.x
    (``tests/test_diagnostico.py``).
    """
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    correct = [sum((_ngrams(p, n) & _ngrams(g, n)).values()) for n in range(1, max_n + 1)]
    total = [max(len(p) - n + 1, 0) for n in range(1, max_n + 1)]
    if correct[0] == 0:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        c, t = correct[n - 1], total[n - 1]
        if n > 1:
            c, t = c + 1, t + 1
        if t == 0:
            break
        logs.append(math.log(c / t))
    bp = 1.0 if len(p) >= len(g) else math.exp(1 - len(g) / len(p))
    return bp * math.exp(sum(logs) / len(logs))


def bleu1_score(pred: str, gold: str) -> float:
    """BLEU-1: precisão de unigramas com a penalidade de brevidade."""
    return bleu_score(pred, gold, max_n=1)


def rougel_score(pred: str, gold: str) -> float:
    """ROUGE-L F1 (Lin, 2004): maior subsequência comum entre predição e gabarito, em tokens normalizados."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    prev = [0] * (len(g) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(g):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec)


def all_scores(pred: str, golds: list[str]) -> dict:
    """EM, F1, Acc, BLEU, BLEU-1 e ROUGE-L, cada um com o máximo sobre as respostas de referência."""
    golds = [g for g in (golds or []) if g]
    em, f1 = best_em_f1(pred, golds)
    best = lambda fn: max((fn(pred, g) for g in golds), default=0.0)  # noqa: E731
    return {"em": em, "f1": round(f1, 4), "acc": contains_answer(pred, golds), "bleu": round(best(bleu_score), 4),
            "bleu1": round(best(bleu1_score), 4), "rougel": round(best(rougel_score), 4)}


# ----------------------------------------------------------------------------- títulos e tokens
def norm_title(t: str) -> str:
    """Título normalizado: deduplicação de documentos e recall de suporte (SR)."""
    t = unicodedata.normalize("NFKC", t or "").lower().strip()
    t = t.strip('"').strip()
    return _WS.sub(" ", t)


def support_recall(titles: list[str], gold_titles: list[str]) -> float:
    """SR: fração dos documentos de suporte anotados que foram recuperados (por título normalizado)."""
    gold = {norm_title(t) for t in gold_titles or [] if t}
    if not gold:
        return 0.0
    got = {norm_title(t) for t in titles or []}
    return len(gold & got) / len(gold)


def content_tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in STOPWORDS]


def approx_words(text: str) -> int:
    return len((text or "").split())


def truncate_words(text: str, max_words: int) -> str:
    w = (text or "").split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " …"


# ----------------------------------------------------------------------------- respostas
_FINAL_PATTERNS = [
    re.compile(r"so the final answer is\s*[:：]?\s*(.+)", re.I | re.S),
    re.compile(r"so the answer is\s*[:：]?\s*(.+)", re.I | re.S),
    re.compile(r"the final answer is\s*[:：]?\s*(.+)", re.I | re.S),
    re.compile(r"final answer\s*[:：]\s*(.+)", re.I | re.S),
    re.compile(r"^\s*answer\s*[:：]\s*(.+)", re.I | re.S | re.M),
]


def clean_short_answer(ans: str) -> str:
    ans = (ans or "").strip()
    ans = ans.splitlines()[0] if ans else ""
    ans = ans.strip().strip("*").strip().strip('"').strip("'").strip()
    ans = re.sub(r"^(the answer is|answer)\s*[:：]\s*", "", ans, flags=re.I)
    return ans.rstrip(".").strip()


def extract_final_answer(raw: str) -> str:
    """Pega a resposta curta depois de 'So the final answer is:' (prefixo usado pelos dois artigos).

    Sem prefixo, cai na primeira linha não vazia (os prompts pedem só a resposta curta).
    """
    raw = raw or ""
    for pat in _FINAL_PATTERNS:
        matches = list(pat.finditer(raw))
        if matches:
            return clean_short_answer(matches[-1].group(1))
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    return clean_short_answer(lines[0]) if lines else ""


def split_official(raw: str) -> str:
    """Extração idêntica ao repositório oficial do RAG-Critic: split('So the final answer is:')[-1]."""
    return clean_short_answer((raw or "").split("So the final answer is:")[-1])


# ----------------------------------------------------------------------------- passagens
def doc_contents(doc: dict) -> str:
    """Formato do corpus wiki do FlashRAG / RAG-Critic: '"Título"\\ntexto'."""
    return f"\"{doc['title']}\"\n{doc['text']}"


def numbered_passages(docs: list[dict], max_words: int | None = 180) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        text = truncate_words(d["text"], max_words) if max_words else d["text"]
        out.append(f"[{i}] {d['title']}: {text}")
    return "\n".join(out)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def first_sentence(text: str) -> str:
    s = split_sentences(text)
    return s[0] if s else (text or "").strip()
