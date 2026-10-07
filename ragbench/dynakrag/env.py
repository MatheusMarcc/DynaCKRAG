"""As operações atômicas do DynaKRAG (Tabela 5) sobre o estado de evidência.

Cada ação recebe um estado e devolve um NOVO estado (sem efeito colateral). Isso é o que permite a
rotulagem offline da Eq. (1): a partir do MESMO estado, todas as ações válidas são executadas e mede-se
a variação do recall de suporte.

Detalhes do artigo seguidos aqui:
  * cada ação que traz documentos acrescenta os 3 primeiros NÃO VISTOS, depois de deduplicar por
    identificador e por título normalizado ("adds the top three unseen documents after identifier- and
    normalized-title-based deduplication");
  * ``retrieve_more`` avança o ranking inicial (a "fronteira") sem chamar o LLM;
  * ``gap_query`` busca com a descrição da informação ausente produzida pela sonda (sem LLM);
  * ``rewrite_query`` reformula a query com a evidência acumulada (1 chamada) e busca;
  * ``bridge_entity_expand`` busca em torno de uma entidade-ponte detectada na evidência;
  * ``sufficiency_check`` pergunta a G se a evidência basta e, se não, registra a lacuna. Padrão: UMA chamada
    (``probe: one_step``) — na Tabela 6 do artigo, com limite de 1 retrieval, são exatamente 3,000 chamadas ao LLM
    por pergunta (sonda + compressão + resposta) e, com limite 2, 3,006–3,100 embora 74–95% das perguntas façam uma
    segunda busca: a descrição da lacuna sai na mesma chamada do julgamento, e a gap_query não chama o LLM;
  * ``compress_answer_evidence`` extrai trechos que se apoiam mutuamente para a resposta final.
"""
from __future__ import annotations

import re
from collections import Counter

from ..core.jsonutil import parse_obj
from ..core.text import extract_final_answer, norm_title, numbered_passages
from . import prompts as P
from .state import STOP, EvidenceState, control_rules, control_signature

_WORDCH = re.compile(r"\w")
_CAP = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+(?:of|the|de|da|do|del|von|van|and|&|la|le|du|di)?\s*[A-Z][\w'’\-]*)*")
_BAD_START = {"The", "A", "An", "In", "On", "He", "She", "It", "They", "His", "Her", "This", "That", "After",
              "Before", "During", "As", "At", "By", "For", "From", "Its", "Their", "There", "When", "While",
              "However", "Although", "Since", "Following", "According", "Born", "These", "Those", "Many", "Some",
              "Most", "Both", "Other", "Also", "Unlike", "With", "Despite", "Upon", "Under", "Between", "Among",
              "If", "Of", "To", "Is", "Was", "We", "You", "I", "One", "Two", "Three", "Its", "Our"}
_SKIP = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
         "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
         "american", "british", "english", "french", "german", "italian", "spanish", "canadian", "australian"}


def detect_bridges(docs: list[dict], question: str, used: list[str], limit: int = 5) -> list[str]:
    """Entidades-ponte: nomes próprios citados na evidência que ainda não têm documento no estado.

    O artigo não especifica o detector. Esta heurística funciona bem nos três datasets: a entidade do
    2º salto costuma ser citada no parágrafo do 1º salto e é o título do parágrafo que falta.
    """
    qn = norm_title(question)
    have = {norm_title(d["title"]) for d in docs} | {norm_title(b) for b in used}
    cnt: Counter = Counter()
    first: dict = {}
    for di, d in enumerate(docs):
        for m in _CAP.finditer(d.get("text", "")):
            parts = m.group(0).strip(" -'’.").split()
            while parts and parts[0] in _BAD_START:
                parts = parts[1:]
            e = " ".join(parts)
            ne = norm_title(e)
            if len(e) < 4 or ne in _SKIP or ne in have or ne in qn:
                continue
            cnt[e] += 1
            first.setdefault(e, (di, m.start()))
    ranked = sorted(cnt, key=lambda e: (-(cnt[e] + 0.5 * (" " in e)), first[e]))
    return ranked[:limit]


def _first_line(text: str, fallback: str, max_words: int = 24) -> str:
    for ln in (text or "").splitlines():
        ln = ln.strip().strip("*").strip().strip('"').strip("'").strip()
        ln = re.sub(r"^(new query|query|search query|missing information)\s*[:：]\s*", "", ln, flags=re.I)
        if ln:
            return " ".join(ln.split()[:max_words])
    return fallback


class DynaEnv:
    def __init__(self, cfg: dict, dataset: str, retriever, llm):
        dk = cfg["dynakrag"]
        self.cfg, self.dk, self.dataset, self.r, self.llm = cfg, dk, dataset, retriever, llm
        self.k = int(dk["docs_per_action"])
        self.depth = int(dk["initial_depth"])
        self.max_final = int(dk["max_final_docs"])
        rules = control_rules(cfg)
        self.probe_mode = rules["probe"]
        self.stop_rule = rules["stop_rule"]          # lidos pela camada de validade
        self.max_probes = rules["max_probes"]
        self.control = control_signature(cfg)        # gravado em cada linha de resultado
        comp = dk["compression"]
        self.comp = {**comp["default"], **(comp.get(dataset) or {})}
        self.answer_max_tokens = int(dk["answer_max_tokens"])
        self.passage_words = int(dk.get("passage_words", 180))
        self._bridge_memo: dict = {}

    # ------------------------------------------------------------------ estado inicial
    def init_state(self, rec: dict) -> EvidenceState:
        ranking = [[self.r.docs[i]["id"], sc] for i, sc in self.r.ranked(rec["question"], self.depth)]
        return EvidenceState(qid=rec["id"], question=rec["question"], query=rec["question"], ranking=ranking)

    # ------------------------------------------------------------------ helpers
    def passages(self, docs: list[dict]) -> str:
        return numbered_passages(docs[: self.max_final], self.passage_words)

    def _add(self, s: EvidenceState, hits: list[dict], via: str) -> int:
        seen_ids, seen_t = s.seen_ids(), s.seen_titles()
        added = 0
        for h in hits:
            nt = norm_title(h["title"])
            if h["id"] in seen_ids or nt in seen_t:
                continue
            s.docs.append({"id": h["id"], "title": h["title"], "text": h["text"], "score": float(h.get("score", 0.0)),
                           "via": via, "step": s.steps})
            seen_ids.add(h["id"])
            seen_t.add(nt)
            added += 1
            if added >= self.k:
                break
        return added

    def _unseen_hits(self, s: EvidenceState, query: str, n: int) -> list[dict]:
        hits = self.r.search(query, k=n + 4 * self.k, exclude_ids=s.seen_ids(), depth=max(60, len(s.docs) + 8 * self.k))
        seen_t = s.seen_titles()
        return [h for h in hits if norm_title(h["title"]) not in seen_t][:n]

    def frontier_has_unseen(self, s: EvidenceState) -> bool:
        seen_ids, seen_t = s.seen_ids(), s.seen_titles()
        for did, _ in s.ranking[s.frontier:]:
            if did not in seen_ids and norm_title(self.r.get(did)["title"]) not in seen_t:
                return True
        return False

    def query_has_unseen(self, s: EvidenceState, query: str) -> bool:
        return bool(query) and bool(self._unseen_hits(s, query, 1))

    def bridge_candidates(self, s: EvidenceState) -> list[str]:
        key = (s.question, tuple(d["id"] for d in s.docs), tuple(s.bridges_used))
        if key not in self._bridge_memo:
            if len(self._bridge_memo) > 50000:
                self._bridge_memo.clear()
            self._bridge_memo[key] = detect_bridges(s.docs, s.question, s.bridges_used)
        return self._bridge_memo[key]

    # ------------------------------------------------------------------ transição
    def step(self, s: EvidenceState, action: str) -> EvidenceState:
        s2 = s.copy()
        getattr(self, action)(s2)
        if action != STOP:
            s2.history.append(action)
        return s2

    # ------------------------------------------------------------------ operações atômicas
    def retrieve_more(self, s: EvidenceState) -> None:
        seen_ids, seen_t = s.seen_ids(), s.seen_titles()
        batch = []
        while s.frontier < len(s.ranking) and len(batch) < self.k:
            did, sc = s.ranking[s.frontier]
            s.frontier += 1
            d = self.r.get(did)
            nt = norm_title(d["title"])
            if did in seen_ids or nt in seen_t:
                continue
            batch.append({**d, "score": sc})
            seen_ids.add(did)
            seen_t.add(nt)
        self._add(s, batch, "retrieve_more")
        s.n_ret += 1

    def gap_query(self, s: EvidenceState) -> None:
        self._add(s, self._unseen_hits(s, s.gap, self.k), "gap_query")
        s.query = s.gap
        s.n_ret += 1

    def rewrite_query(self, s: EvidenceState) -> None:
        out = self.llm.chat("generator", [
            {"role": "system", "content": P.REWRITE_SYSTEM},
            {"role": "user", "content": P.REWRITE_USER.format(question=s.question, query=s.query,
                                                              passages=self.passages(s.docs))},
        ], task="dk_rewrite", max_tokens=48)
        q = _first_line(out, s.question)
        self._add(s, self._unseen_hits(s, q, self.k), "rewrite_query")
        s.query = q
        s.n_ret += 1
        s.n_llm += 1

    def bridge_entity_expand(self, s: EvidenceState) -> None:
        cands = self.bridge_candidates(s)
        if not cands:
            return
        e = cands[0]
        self._add(s, self._unseen_hits(s, e, self.k), "bridge_entity_expand")
        s.bridges_used.append(e)
        s.n_ret += 1

    def sufficiency_check(self, s: EvidenceState) -> None:
        passages = self.passages(s.docs)
        user = P.SUFF_USER.format(question=s.question, passages=passages)
        if self.probe_mode == "one_step":
            out = self.llm.chat("generator", [{"role": "system", "content": P.SUFF_ONE_SYSTEM},
                                              {"role": "user", "content": user}],
                                task="dk_suff_judge", max_tokens=96, fmt=P.SUFF_ONE_SCHEMA)
            obj = parse_obj(out, {}) or {}
            ok = bool(obj.get("sufficient", False)) if isinstance(obj, dict) else False
            gap = _first_line(str(obj.get("missing") or "") if isinstance(obj, dict) else "", "")
            s.n_llm += 1
        else:
            out = self.llm.chat("generator", [{"role": "system", "content": P.SUFF_SYSTEM},
                                              {"role": "user", "content": user}],
                                task="dk_suff_judge", max_tokens=24, fmt=P.SUFF_SCHEMA)
            obj = parse_obj(out, {}) or {}
            ok = bool(obj.get("sufficient", False)) if isinstance(obj, dict) else False
            s.n_llm += 1
            gap = ""
            if not ok:
                g = self.llm.chat("generator", [{"role": "system", "content": P.GAP_SYSTEM},
                                                {"role": "user", "content": P.GAP_USER.format(question=s.question, passages=passages)}],
                                  task="dk_gap", max_tokens=40)
                gap = _first_line(g, "")
                s.n_llm += 1
        s.suff = int(ok)
        s.docs_at_check = len(s.docs)
        s.n_checks += 1
        if ok:
            s.gap = ""
        else:
            s.gap = gap or s.gap
            if gap:
                s.gaps_seen.append(gap)

    def stop_answer(self, s: EvidenceState) -> None:  # noqa: D401 — termina a aquisição
        return None

    # ------------------------------------------------------------------ término
    def compress(self, s: EvidenceState, docs: list[dict]) -> tuple[str, str]:
        n, mt = int(self.comp["max_snippets"]), int(self.comp["max_tokens"])
        system = (P.COMPRESS_TABLE_SYSTEM if self.comp.get("style") == "fact_table" else P.COMPRESS_EXTRACT_SYSTEM).format(n=n)
        raw = self.llm.chat("generator", [{"role": "system", "content": system},
                                          {"role": "user", "content": P.COMPRESS_USER.format(question=s.question,
                                                                                             passages=self.passages(docs))}],
                            task="dk_compress", max_tokens=mt)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().lower().startswith(("output", "snippets"))]
        # linhas sem texto nenhum ("- - -", "| |") não são evidência: saem. Se não sobrar nada, vale a regra que já
        # existia para a saída vazia: a resposta é gerada com os documentos sem compressão.
        lines = [ln for ln in lines if len(_WORDCH.findall(ln)) >= 3][:n]
        return ("\n".join(lines) if lines else self.passages(docs)), raw

    def answer(self, question: str, evidence: str, task: str = "dk_answer") -> tuple[str, str]:
        raw = self.llm.chat("generator", [{"role": "system", "content": P.ANSWER_SYSTEM},
                                          {"role": "user", "content": P.ANSWER_USER.format(evidence=evidence or "(no evidence)",
                                                                                           question=question)}],
                            task=task, max_tokens=self.answer_max_tokens)
        return extract_final_answer(raw), raw

    def answer_no_retrieval(self, question: str) -> tuple[str, str]:
        raw = self.llm.chat("generator", [{"role": "system", "content": P.NO_RETRIEVAL_SYSTEM},
                                          {"role": "user", "content": P.NO_RETRIEVAL_USER.format(question=question)}],
                            task="no_retrieval_answer", max_tokens=self.answer_max_tokens)
        return extract_final_answer(raw), raw
