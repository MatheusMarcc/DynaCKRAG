"""Laço B (RAG-Critic): planejar e executar o reparo das respostas que o crítico marcou como erro.

Três modos, para você comparar:
  * code  — fiel ao artigo: o planner escreve Python com 5 funções e um executor roda o código.
            ATENÇÃO: o repositório oficial usa exec() sem restrição; aqui o código passa por
            uma whitelist de AST antes de rodar (sem imports, sem atributos, sem laços).
  * menu  — o planner escolhe passos de um menu fixo via JSON schema (structured outputs do
            Ollama). Funciona muito melhor com modelos de 3B do que gerar código livre.
  * state — integração nova: a crítica vira uma lacuna no estado do DynaKRAG e o MESMO
            controlador do laço A decide o reparo, com alguns passos extras.
"""
from __future__ import annotations

import ast
import os

import json_repair
from tqdm import tqdm

from . import prompts as P
from .loop import final_context, generate_answer, run_episode
from .metrics import best_em_f1, support_recall
from .state import EvidenceState
from .utils import numbered_passages, passage_str, read_jsonl, write_jsonl


class BudgetExceeded(RuntimeError):
    pass


# ------------------------------------------------------------------ caixa de ferramentas
class Toolbox:
    """As 5 funções do RAG-Critic, ligadas ao nosso retriever/LLM e com orçamento de retrieval."""

    def __init__(self, env, question: str, max_retrievals: int):
        self.env, self.question = env, question
        self.left = max_retrievals
        self.used = 0
        self.llm_left = 12            # evita que um plano faça dezenas de chamadas (ex.: comprehension aninhada)
        self.k = env.cfg["retriever"]["docs_per_action"]

    def _llm_budget(self):
        if self.llm_left <= 0:
            raise BudgetExceeded("orçamento de chamadas ao LLM do reparo esgotado")
        self.llm_left -= 1

    def Retrieval(self, query: str, topk: int = 3) -> list[str]:
        if self.left <= 0:
            raise BudgetExceeded("orçamento de retrieval do reparo esgotado")
        self.left -= 1
        self.used += 1
        return [passage_str(h) for h in self.env.r.search(str(query), max(1, min(int(topk), 5)))]

    def RewriteQuery(self, query: str, instruction: str = "clarify") -> list[str]:
        self._llm_budget()
        q = self.env.llm.chat("planner", [
            {"role": "system", "content": f"Rewrite the search query ({instruction}: make it more specific and "
                                          "searchable). Output only the query."},
            {"role": "user", "content": str(query)}], task="repair_rewrite", max_tokens=48)
        return [(q.strip().splitlines() or [str(query)])[0].strip().strip('"') or str(query)]

    def DecomposeQuery(self, query: str) -> list[str]:
        self._llm_budget()
        out = self.env.llm.chat("planner", [
            {"role": "system", "content": P.DECOMPOSE_SYSTEM},
            {"role": "user", "content": str(query)}], task="repair_decompose", max_tokens=128,
            fmt=P.DECOMPOSE_SCHEMA)
        j = json_repair.loads(out)
        subs = j.get("subquestions", []) if isinstance(j, dict) else (j if isinstance(j, list) else [])
        return [str(x) for x in subs if str(x).strip()][:3] or [str(query)]

    def RefineDoc(self, query: str, doc: str, instruction: str = "summarize") -> str:
        system = P.REFINE_SYSTEM.get(instruction)
        if not system:
            return str(doc)
        self._llm_budget()
        return self.env.llm.chat("generator", [
            {"role": "system", "content": system},
            {"role": "user", "content": P.REFINE_USER.format(question=query, doc=doc)}],
            task="repair_refine", max_tokens=160).strip() or str(doc)

    def GenerateAnswer(self, query: str, docs: list, additional_instruction: str | None = None) -> str:
        self._llm_budget()
        evidence = "\n".join(f"[{i}] {d}" for i, d in enumerate(docs, 1))
        pred, _ = generate_answer(self.env, str(query), evidence, tips=additional_instruction)
        return pred


# ------------------------------------------------------------------ sandbox para o modo code
_ALLOWED_NODES = (ast.Module, ast.Assign, ast.Expr, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant,
                  ast.List, ast.Tuple, ast.ListComp, ast.comprehension, ast.Subscript, ast.Slice, ast.keyword,
                  ast.BinOp, ast.Add, ast.JoinedStr, ast.FormattedValue, ast.UnaryOp, ast.USub)
_FUNCS = {"Retrieval", "RewriteQuery", "DecomposeQuery", "RefineDoc", "GenerateAnswer"}
_BUILTINS = {"len": len, "str": str}


def validate_code(code: str) -> ast.Module:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"construção não permitida: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in (_FUNCS | set(_BUILTINS)):
                raise ValueError(f"chamada não permitida: {ast.unparse(node.func)}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("nome proibido")
    return tree


def extract_code(text: str) -> str:
    if "```" in text:
        text = text.split("```python")[-1] if "```python" in text else text.split("```")[1]
        text = text.split("```")[0]
    return text.strip()


# ------------------------------------------------------------------ modos
def repair_code(env, s: EvidenceState, pred: str, crit: dict) -> tuple[str, dict]:
    docs = s.docs[: env.cfg["loop"]["max_final_docs"]]
    out = env.llm.chat("planner", [
        {"role": "system", "content": P.PLANNER_CODE_SYSTEM},
        {"role": "user", "content": P.PLANNER_CODE_USER.format(
            question=s.question, n_docs=len(docs), titles=[d["title"] for d in docs], pred=pred,
            tags=crit["tag1"] + crit["tag2"], analysis=crit["analysis"][:600], hints=P.error_hints(crit["tag1"]))},
    ], task="planner_code", max_tokens=400)
    code = extract_code(out)
    tb = Toolbox(env, s.question, env.cfg["repair"]["max_retrievals"])
    ns = {"__builtins__": _BUILTINS, "question": s.question, "doc_list": [passage_str(d) for d in docs],
          "previous_pred": pred, **{f: getattr(tb, f) for f in _FUNCS}}
    try:
        exec(compile(validate_code(code), "<plan>", "exec"), ns)  # noqa: S102 — código validado por whitelist
        new = ns.get("final_answer")
        if not isinstance(new, str) or not new.strip():
            raise ValueError("o código não definiu final_answer")
        return new, {"plan": code, "ok": True, "repair_retrievals": tb.used}
    except Exception as e:  # fallback: responde de novo usando a crítica como instrução
        tb.llm_left = max(tb.llm_left, 1)
        new = tb.GenerateAnswer(s.question, ns["doc_list"], crit["analysis"][:400] or None)
        return new, {"plan": code, "ok": False, "error": f"{type(e).__name__}: {e}", "repair_retrievals": tb.used}


def repair_menu(env, s: EvidenceState, pred: str, crit: dict) -> tuple[str, dict]:
    docs = s.docs[: env.cfg["loop"]["max_final_docs"]]
    out = env.llm.chat("planner", [
        {"role": "system", "content": P.PLANNER_MENU_SYSTEM},
        {"role": "user", "content": P.PLANNER_MENU_USER.format(
            question=s.question, passages=numbered_passages(docs, 400), pred=pred,
            tags=crit["tag1"] + crit["tag2"], analysis=crit["analysis"][:600], hints=P.error_hints(crit["tag1"]))},
    ], task="planner_menu", max_tokens=256, fmt=P.PLANNER_MENU_SCHEMA)
    plan = json_repair.loads(out)
    plan = plan if isinstance(plan, dict) else {}
    tb = Toolbox(env, s.question, env.cfg["repair"]["max_retrievals"])
    pool = [passage_str(d) for d in docs]
    done = []
    for st in (plan.get("steps") or [])[:3]:
        a, arg = str(st.get("action", "")), str(st.get("argument", "") or "")
        try:
            if a == "retrieve_same":
                pool = tb.Retrieval(s.question, 3) + pool
            elif a == "retrieve_new":
                pool = tb.Retrieval(arg or tb.RewriteQuery(s.question)[0], 3) + pool
            elif a == "decompose":
                for sq in tb.DecomposeQuery(s.question)[:2]:
                    pool = tb.Retrieval(sq, 2) + pool
            elif a == "summarize_docs":
                pool = [tb.RefineDoc(s.question, d, "summarize") for d in pool[:5]]
            elif a == "drop_docs":
                drop = {int(x) - 1 for x in arg.replace(" ", "").split(",") if x.isdigit()}
                kept = [d for i, d in enumerate(pool) if i not in drop]
                pool = kept or pool
            else:
                continue
            done.append(a)
        except BudgetExceeded:
            break
    seen, uniq = set(), []
    for d in pool:
        if d not in seen:
            seen.add(d); uniq.append(d)
    instr = str(plan.get("answer_instruction") or crit["analysis"][:300] or "")
    tb.llm_left = max(tb.llm_left, 1)
    new = tb.GenerateAnswer(s.question, uniq[: env.cfg["loop"]["max_final_docs"]], instr or None)
    return new, {"plan": plan, "executed": done, "ok": bool(done), "repair_retrievals": tb.used}


def repair_state(env, policy, s: EvidenceState, pred: str, crit: dict) -> tuple[str, dict]:
    gap = env.llm.chat("generator", [
        {"role": "system", "content": P.GAP_FROM_CRITIQUE_SYSTEM},
        {"role": "user", "content": P.GAP_FROM_CRITIQUE_USER.format(question=s.question,
                                                                    analysis=crit["analysis"][:600])},
    ], task="gap_from_critique", max_tokens=48)
    s = s.copy()
    n_hist, n_ret0 = len(s.history), s.n_ret
    s.gap = (gap.strip().splitlines() or [""])[0].strip().strip('"') or s.question
    s.gap_used, s.last_suff = False, 0
    s.docs_at_last_check = len(s.docs)
    s.step_budget = len(s.history) + env.cfg["repair"]["extra_steps"]
    s.ret_budget = s.n_ret + env.cfg["repair"]["max_retrievals"]
    s = run_episode(env, policy, s, env.cfg["loop"]["max_final_docs"])
    _, evidence = final_context(env, s, env.cfg["loop"]["compression"])
    new, _ = generate_answer(env, s.question, evidence, tips=crit["analysis"][:300] or None)
    return new, {"gap": s.gap, "extra_history": s.history[n_hist:], "state": s.to_dict(),
                 "repair_retrievals": s.n_ret - n_ret0}


# ------------------------------------------------------------------ orquestração
def run_repair(cfg: dict, env, run_name: str, mode: str, policy=None) -> str:
    run_dir = os.path.join(cfg["paths"]["runs"], run_name)
    preds = read_jsonl(os.path.join(run_dir, "preds.jsonl"))
    crits = {c["qid"]: c for c in read_jsonl(os.path.join(run_dir, "critic.jsonl"))}
    gold_sup = {r["qid"]: r["supporting_titles"] for r in read_jsonl(cfg["paths"]["dev"])}
    rows = []
    for r in tqdm(preds, desc=f"reparo [{mode}]"):
        c = crits.get(r["qid"], {"judgement": "unknown"})
        s = EvidenceState.from_dict(r["state"])
        usage = {"calls": r["usage"]["calls"] + c.get("usage", {}).get("calls", 0),
                 "prompt_tokens": r["usage"]["prompt_tokens"] + c.get("usage", {}).get("prompt_tokens", 0),
                 "completion_tokens": r["usage"]["completion_tokens"] + c.get("usage", {}).get("completion_tokens", 0)}
        info, new = {"repaired": False}, r["pred"]
        if c["judgement"] == "error":
            with env.llm.track() as u:
                if mode == "code":
                    new, info = repair_code(env, s, r["pred"], c)
                elif mode == "menu":
                    new, info = repair_menu(env, s, r["pred"], c)
                elif mode == "state":
                    new, info = repair_state(env, policy, s, r["pred"], c)
                else:
                    raise ValueError(mode)
            info["repaired"] = True
            for k in ("calls", "prompt_tokens", "completion_tokens"):
                usage[k] += u[k]
        em, f1 = best_em_f1(new, r["answers"])
        titles = [d["title"] for d in info.get("state", r["state"])["docs"]]
        rows.append({"qid": r["qid"], "question": r["question"], "answers": r["answers"], "pred": new,
                     "pred_before": r["pred"], "em_before": r["em"], "f1_before": r["f1"], "em": em, "f1": f1,
                     "sr": support_recall(titles, gold_sup.get(r["qid"], [])), "critic": c.get("judgement"),
                     "n_ret": r["n_ret"] + info.get("repair_retrievals", 0),
                     "steps": r["steps"] + len(info.get("extra_history", [])),
                     "usage": usage, "info": {k: v for k, v in info.items() if k != "state"}})
    path = os.path.join(run_dir, f"repaired_{mode}.jsonl")
    write_jsonl(path, rows)
    return path
