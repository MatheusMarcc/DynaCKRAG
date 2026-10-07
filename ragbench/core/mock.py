"""LLM falso e determinístico para testar os pipelines sem GPU e sem Ollama (perfil ``smoke``).

Não tenta ser inteligente: produz saídas no FORMATO que cada etapa espera (JSON, código, prefixo
"So the final answer is:"), com um leitor heurístico simples para as respostas. O objetivo é
exercitar todos os caminhos de código (parsers, validade, sandbox, métricas, relatórios).
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from .text import content_tokens

_CAP = re.compile(r"\b[A-Z][\w'\-]*(?:\s+(?:of|the|de|van|von)?\s*[A-Z][\w'\-]*)*")
_YESNO = ("is ", "are ", "was ", "were ", "did ", "does ", "do ", "can ", "has ", "have ", "could ")


def _h(*parts) -> int:
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _all_text(messages: list[dict]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def _find(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.S | re.I)
    return m.group(1).strip() if m else default


def _question(text: str) -> str:
    for pat in [r"\*\*Question\*\*:\s*(.+?)\s*\*\*Response", r"Question:\s*(.+?)(?:\n|$)",
                r'question = "(.+?)"', r"Original query:\s*(.+?)(?:\n|$)"]:
        q = _find(pat, text)
        if q:
            return q
    return text.strip().splitlines()[-1] if text.strip() else ""


def _toy_answer(q: str, evidence: str) -> str | None:
    """Leitor "perfeito" para os modelos de pergunta do dataset sintético (só acerta se a evidência tiver os fatos)."""
    m = re.match(r"In which city was the director of (.+?) born\?", q)
    if m:
        d = re.search(re.escape(m.group(1)) + r" is a \d+ drama film directed by ([A-Z][\w\-]+ [A-Z][\w\-]+)", evidence)
        if d:
            c = re.search(re.escape(d.group(1)) + r" was born in ([A-Z][\w]+(?: [A-Z][\w]+)*)", evidence)
            if c:
                return c.group(1)
        return "unknown"
    m = re.match(r"Who was born first, (.+?) or (.+?)\?", q)
    if m:
        a, b = m.group(1), m.group(2)
        ya = re.search(re.escape(a) + r" \(born (\d{4})\)", evidence)
        yb = re.search(re.escape(b) + r" \(born (\d{4})\)", evidence)
        if ya and yb:
            return a if int(ya.group(1)) < int(yb.group(1)) else b
        return "unknown"
    m = re.match(r"Who directed the film (.+?)\?", q)
    if m:
        d = re.search(re.escape(m.group(1)) + r" is a \d+ drama film directed by ([A-Z][\w\-]+ [A-Z][\w\-]+)", evidence)
        return d.group(1) if d else "unknown"
    return None


def toy_gap(q: str, evidence: str) -> str | None:
    m = re.match(r"In which city was the director of (.+?) born\?", q)
    if m:
        d = re.search(re.escape(m.group(1)) + r" is a \d+ drama film directed by ([A-Z][\w\-]+ [A-Z][\w\-]+)", evidence)
        return f"where was {d.group(1)} born" if d else f"director of the film {m.group(1)}"
    m = re.match(r"Who was born first, (.+?) or (.+?)\?", q)
    if m:
        for p in (m.group(1), m.group(2)):
            if not re.search(re.escape(p) + r" \(born", evidence):
                return f"{p} birth year"
    return None


def read_answer(question: str, evidence: str) -> str:
    """Leitor heurístico: frase com maior sobreposição com a pergunta → entidade que não está nela."""
    q = question.strip()
    toy = _toy_answer(q, evidence)
    if toy is not None:
        return toy
    if q.lower().startswith(_YESNO):
        return "yes"
    qtok = set(content_tokens(q))
    best, best_s = "", -1
    for sent in re.split(r"(?<=[.!?])\s+|\n", evidence):
        s = len(qtok & set(content_tokens(sent)))
        if s > best_s:
            best, best_s = sent, s
    qlow = q.lower()
    for m in _CAP.finditer(best):
        cand = m.group(0).strip()
        if len(cand) > 2 and cand.lower() not in qlow and not cand.lower().startswith(("the ", "passage", "question")):
            return cand
    nums = re.findall(r"\b\d{3,4}\b", best)
    if nums:
        return nums[0]
    return "unknown"


class MockBrain:
    def __init__(self, seed: int = 13):
        self.seed = seed

    # ------------------------------------------------------------------ roteamento por tarefa
    def respond(self, task: str, messages: list[dict], fmt=None, salt=None) -> str:
        text = _all_text(messages)
        user = _last_user(messages)
        q = _question(user) or _question(text)
        h = _h(self.seed if salt is None else salt, task, text[:2000])
        fn = getattr(self, "_" + task, None)
        if fn is not None:
            return fn(text=text, user=user, q=q, h=h)
        # padrão: resposta curta com o prefixo usado pelos dois artigos
        return f"So the final answer is: {read_answer(q, text)}"

    def logprobs(self, text: str) -> list[dict]:
        toks = text.split()
        return [{"token": t, "logprob": math.log(0.3 if _h(t) % 5 == 0 else 0.9)} for t in toks]

    # ------------------------------------------------------------------ DynaKRAG
    def _dk_answer(self, text, user, q, h):
        return read_answer(q, text)

    _no_retrieval_answer = _dk_answer
    _sub_answer = _dk_answer

    def _dk_suff_judge(self, text, user, q, h):
        ok = read_answer(q, text) != "unknown"
        if "'missing'" in text:              # sonda de uma chamada: o julgamento e a lacuna saem juntos
            return json.dumps({"sufficient": ok, "missing": "" if ok else self._dk_gap(text, user, q, h)})
        return json.dumps({"sufficient": ok})

    def _dk_gap(self, text, user, q, h):
        g = toy_gap(q, text)
        if g:
            return g
        ents = [m.group(0) for m in _CAP.finditer(text) if m.group(0).lower() not in q.lower()]
        return f"information about {ents[h % len(ents)]}" if ents else f"more details about {q}"

    def _dk_rewrite(self, text, user, q, h):
        ents = [m.group(0) for m in _CAP.finditer(text) if m.group(0).lower() not in q.lower() and len(m.group(0)) > 3]
        return (q + " " + ents[h % len(ents)]) if ents else q

    def _dk_compress(self, text, user, q, h):
        qtok = set(content_tokens(q))
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if len(s.split()) > 3]
        sents.sort(key=lambda s: -len(qtok & set(content_tokens(s))))
        return "\n".join("- " + s for s in sents[:5])

    # ------------------------------------------------------------------ baselines multi-hop
    def _ircot_step(self, text, user, q, h):
        if text.count("[") >= 6 or h % 3 == 0:
            return f"So the answer is: {read_answer(q, text)}."
        return f"{read_answer(q, text)} is mentioned in the documents."

    def _selfask_step(self, text, user, q, h):
        if user.count("Intermediate answer:") >= 2 or h % 4 == 0:
            return f"So the final answer is: {read_answer(q, text)}"
        ents = [m.group(0) for m in _CAP.finditer(text[-1500:])]
        tgt = ents[h % len(ents)] if ents else "the subject"
        return f"Follow up: Who or what is {tgt}?"

    def _crag_eval(self, text, user, q, h):
        n = len(re.findall(r"^\[\d+\]", text, re.M))
        labels = ["correct" if (_h(h, i) % 3 == 0) else ("ambiguous" if _h(h, i) % 3 == 1 else "incorrect") for i in range(n)]
        return json.dumps({"labels": labels})

    def _crag_refine(self, text, user, q, h):
        return self._dk_compress(text, user, q, h)

    def _crag_rewrite(self, text, user, q, h):
        return " ".join(content_tokens(q)[:6])

    def _adaptive_classify(self, text, user, q, h):
        return json.dumps({"label": "ABC"[h % 3]})

    def _s2g_judge(self, text, user, q, h):
        if h % 3 == 0 or text.count("[") > 8:
            return json.dumps({"sufficient": True, "known_facts": ["fact"], "gaps": []})
        return json.dumps({"sufficient": False, "known_facts": [], "gaps": [{"missing": "missing fact", "query": self._dk_gap(text, user, q, h)}]})

    def _corag_subquery(self, text, user, q, h):
        return self._dk_gap(text, user, q, h)

    def _par2_decompose(self, text, user, q, h):
        words = content_tokens(q)
        return json.dumps({"subquestions": [" ".join(words[:4]) or q, " ".join(words[2:7]) or q]})

    def _par2_followup(self, text, user, q, h):
        return self._dk_gap(text, user, q, h)

    # ------------------------------------------------------------------ RAG-Critic
    def _rc_generate(self, text, user, q, h):
        return f"Based on the passages, the answer follows. So the final answer is: {read_answer(q, text)}"

    _rc_generate_answer = _rc_generate
    _flare_draft = _rc_generate
    _selfrefine_refine = _rc_generate

    def _critic_output(self, text, h):
        pred = _find(r"Model's Prediction:\s*(.+?)\n\s*Please first determine", text)
        short = pred.split("So the final answer is:")[-1].strip().rstrip(".")
        passages = text.split("Model's Prediction:")[0]
        if short and short.lower() != "unknown" and short.lower() in passages.lower() and h % 4 != 0:
            return "{'Judgement': 'Correct'}"
        t1 = ["Incomplete Information", "Irrelevant Information", "Inaccurate or Misunderstood Response",
              "Incomplete or Missing Response"][h % 4]
        t2 = ["Insufficient or Incomplete Information Retrieval", "Irrelevant Information Retrieval",
              "Factual Inaccuracies", "Omission of Key Information"][h % 4]
        return str({"Judgement": "Error", "Error_analysis": "The prediction is not supported by the passages.",
                    "tag1": [t1], "tag2": [t2], "tag3": ["Mock Error"]})

    def _rc_critic(self, text, user, q, h):
        return self._critic_output(text, h)

    def _rc_critic_zeroshot(self, text, user, q, h):
        out = self._critic_output(text, h + 1)
        return out.replace("'", '"')

    _rc_critic_bench = _rc_critic
    _rc_critic_zeroshot_bench = _rc_critic_zeroshot

    def _rc_plan(self, text, user, q, h):
        variants = [
            'clarified_query = RewriteQuery(query=question, instruction="clarify")\n'
            'retrieved_documents = Retrieval(query=clarified_query[0], topk=5)\n'
            'summarized_documents = [RefineDoc(query=question, doc=doc, instruction="summarize") for doc in retrieved_documents]\n'
            'final_answer = GenerateAnswer(query=question, docs=summarized_documents, additional_instruction="Be precise.")',
            'sub_queries = DecomposeQuery(query=question)\n'
            'docs = doc_list[:2]\n'
            'for sq in sub_queries:\n'
            '    docs.extend(Retrieval(query=sq, topk=3))\n'
            'final_answer = GenerateAnswer(query=question, docs=docs)',
            'new_docs = Retrieval(query=question, topk=5)\n'
            'final_answer = GenerateAnswer(query=question, docs=doc_list + new_docs, additional_instruction="Answer concisely.")',
        ]
        return "```python\n" + variants[h % len(variants)] + "\n```"

    def _rc_rewrite_clarify(self, text, user, q, h):
        return json.dumps({"query": q + " (specific)"})

    _rc_rewrite_custom = _rc_rewrite_clarify

    def _rc_rewrite_expand(self, text, user, q, h):
        return json.dumps([q, q + " details"])

    def _rc_decompose(self, text, user, q, h):
        words = content_tokens(q)
        return json.dumps([" ".join(words[:4]) or q, " ".join(words[3:8]) or q])

    def _rc_refine_summarize(self, text, user, q, h):
        doc = _find(r"Document:\s*(.+?)\nQuestion:", user) or user
        return json.dumps({"refined_document": doc[:300]})

    def _rc_refine_explain(self, text, user, q, h):
        doc = _find(r"Original document:\s*(.+?)\nQuestion:", user) or user
        return json.dumps({"explanation": doc[:300]})

    def _selfrefine_feedback(self, text, user, q, h):
        return "NO ISSUES" if h % 2 == 0 else "The answer may be incomplete; check the passages again."

    def _metarag_internal(self, text, user, q, h):
        return "So the final answer is: " + ("unknown" if h % 2 else read_answer(q, text))

    def _metarag_eval(self, text, user, q, h):
        return json.dumps({"docs_sufficient": h % 3 != 0, "conflict": h % 5 == 0, "reasoning_error": h % 4 == 0})

    def _metarag_rewrite(self, text, user, q, h):
        return q + " background"

    # ------------------------------------------------------------------ sistema de erros (etapas 1–3)
    def _es_judge(self, text, user, q, h):
        if h % 2 == 0:
            return "{'Judgement': 'Correct', 'Error analysis': 'None'}"
        return ("{'Judgement': 'incorrect', 'Error analysis': \"The model's prediction is incorrect because it relied on "
                f"an irrelevant passage when answering: {q[:80]}\"}}")

    def _es_tag(self, text, user, q, h):
        pool = ["Irrelevant Information", "Incomplete Information", "Factual Inaccuracy", "Entity Confusion",
                "Date Mismatch", "Inference Beyond Evidence", "Missing Context", "Overly Verbose"]
        a, b = pool[h % len(pool)], pool[(h // 7) % len(pool)]
        return json.dumps([{"tag": a, "explanation": "mock"}, {"tag": b, "explanation": "mock"}])

    def _es_cluster_name(self, text, user, q, h):
        return ["Retrieval Gaps", "Irrelevant Retrieval", "Factual Errors", "Reasoning Errors",
                "Verbosity", "Entity Confusion"][h % 6]

    def _es_tier1(self, text, user, q, h):
        return "Incomplete Information"
