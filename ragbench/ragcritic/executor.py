"""Executor do programa gerado pelo planner (as 5 funções da Tabela 2, com os prompts oficiais).

O repositório oficial roda o código do planner com ``exec()`` sem nenhuma restrição. Aqui o programa
passa antes por uma **whitelist de AST**: sem import, sem def/lambda/classe, sem while/try/with, sem
nomes ou atributos "dunder", atributos só de métodos seguros de str/list/dict, e um orçamento de chamadas
às funções. O que um planner honesto escreve (atribuições, chamadas, listas, compreensões, for, if,
índices, f-strings) continua funcionando.

Diferenças deliberadas em relação ao código oficial (todas configuráveis/documentadas):
  * ``GenerateAnswer``: o oficial monta a dica ``additional_instruction`` mas NÃO a formata (a string
    literal "{additional_instruction}" vai para o modelo). Aqui a dica é formatada.
  * ``RewriteQuery`` com instrução livre: o prompt oficial tem chaves JSON sem escape e o ``.format``
    quebra. Aqui as chaves estão escapadas.
  * Coerção de tipos (lista onde se espera string, índice onde se espera documento etc.) para que um
    erro de tipo bobo de um planner pequeno não derrube o programa inteiro.
"""
from __future__ import annotations

import ast
import re

from ..core.jsonutil import as_str_list, parse_obj
from ..core.text import doc_contents
from . import prompts as P

FUNCS = ("Retrieval", "RewriteQuery", "DecomposeQuery", "RefineDoc", "GenerateAnswer")
ALIASES = {
    "Retrieval": "Retrieval", "Retrieve": "Retrieval", "retrieval": "Retrieval", "retrieve": "Retrieval",
    "RewriteQuery": "RewriteQuery", "rewrite_query": "RewriteQuery", "Rewrite": "RewriteQuery",
    "DecomposeQuery": "DecomposeQuery", "decompose_query": "DecomposeQuery", "Decompose": "DecomposeQuery",
    "RefineDoc": "RefineDoc", "RefineDocument": "RefineDoc", "refine_doc": "RefineDoc", "Refine": "RefineDoc",
    "GenerateAnswer": "GenerateAnswer", "generate_answer": "GenerateAnswer", "Generate": "GenerateAnswer",
}
SAFE_METHODS = {"append", "extend", "join", "strip", "lstrip", "rstrip", "split", "replace", "lower", "upper",
                "title", "startswith", "endswith", "get", "items", "keys", "values", "insert", "index",
                "count", "pop", "copy"}   # sem .format: format strings acessam atributos "dunder"
ALLOWED = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name, ast.Load, ast.Store,
           ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp,
           ast.GeneratorExp, ast.comprehension, ast.Call, ast.keyword, ast.Subscript, ast.Slice, ast.BinOp,
           ast.Add, ast.Sub, ast.Mod, ast.Div, ast.FloorDiv, ast.UnaryOp, ast.USub, ast.UAdd, ast.Not,
           ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
           ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.If, ast.IfExp, ast.For, ast.Pass, ast.Break, ast.Continue,
           ast.JoinedStr, ast.FormattedValue, ast.Attribute, ast.Starred)
if hasattr(ast, "Index"):  # Python < 3.9
    ALLOWED = ALLOWED + (ast.Index,)  # type: ignore[attr-defined]


class ProgramError(Exception):
    pass


class BudgetExceeded(ProgramError):
    pass


def _safe_range(*args):
    r = range(*[int(a) for a in args])
    if len(r) > 50:
        raise ProgramError("range grande demais")
    return r


SAFE_BUILTINS = {"len": len, "range": _safe_range, "enumerate": enumerate, "zip": zip, "min": min, "max": max,
                 "sorted": sorted, "list": list, "str": str, "int": int, "dict": dict, "set": set, "any": any,
                 "all": all, "reversed": reversed, "sum": sum, "print": lambda *a, **k: None, "True": True,
                 "False": False, "None": None, "isinstance": isinstance, "tuple": tuple, "float": float}


_HARMLESS_IMPORT = re.compile(r"^\s*(from\s+typing\s+import\s+.*|import\s+typing.*)$", re.M)


def clean_code(code: str) -> str:
    """Remove imports inofensivos de ``typing`` (planners pequenos às vezes os escrevem) e cercas soltas."""
    code = _HARMLESS_IMPORT.sub("", code or "")
    return code.replace("```python", "").replace("```", "").strip()


def validate(code: str) -> ast.Module:
    try:
        tree = ast.parse(clean_code(code))
    except SyntaxError as e:
        raise ProgramError(f"SyntaxError: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ProgramError(f"construção não permitida: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ProgramError(f"nome não permitido: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in SAFE_METHODS:
                raise ProgramError(f"atributo não permitido: .{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn not in ALIASES and fn not in SAFE_BUILTINS:
                raise ProgramError(f"função não permitida: {fn}")
    return tree


MAX_ITER = 50


def _guard(iterable):
    """Materializa o iterável (cópia) e limita o tamanho: impede laço infinito em lista que cresce."""
    items = list(iterable)
    if len(items) > MAX_ITER:
        items = items[:MAX_ITER]
    return items


class _GuardLoops(ast.NodeTransformer):
    """Envolve o iterável de todo ``for`` e de toda compreensão com ``__guard__(...)``."""

    def _wrap(self, it):
        return ast.Call(func=ast.Name(id="__guard__", ctx=ast.Load()), args=[it], keywords=[])

    def visit_For(self, node):
        self.generic_visit(node)
        node.iter = self._wrap(node.iter)
        return node

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.iter = self._wrap(node.iter)
        return node


def guard_loops(tree: ast.Module) -> ast.Module:
    return ast.fix_missing_locations(_GuardLoops().visit(tree))


def final_answer_var(code: str) -> str:
    """Mesma heurística do executor oficial (_fetch_final_answer_var_name)."""
    var = "answer"
    for line in code.split("\n")[::-1]:
        if line.startswith("#") or "print" in line:
            continue
        if "GenerateAnswer" in line or "generate_answer" in line:
            var = line.split("=")[0].strip()
            break
    return var


def _flatten(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [str(v) for v in x.values()]
    out = []
    for i in x:
        out.extend(_flatten(i))
    return out


class Executor:
    def __init__(self, llm, retriever, max_calls: int = 12, max_topk: int = 10, max_docs: int = 10,
                 model: str = "generator"):
        self.llm, self.r = llm, retriever
        self.max_calls, self.max_topk, self.max_docs, self.model = max_calls, max_topk, max_docs, model

    # ------------------------------------------------------------------ as 5 funções
    def _mk(self, ctx: dict):
        llm, model = self.llm, self.model

        def tick(name: str):
            ctx["calls"].append(name)
            if len(ctx["calls"]) > self.max_calls:
                raise BudgetExceeded(f"mais de {self.max_calls} chamadas de função")

        def as_query(q) -> str:
            if isinstance(q, (list, tuple)):
                q = next((x for x in _flatten(q) if str(x).strip()), "")
            q = str(q or "").strip()
            return q or ctx["question"]

        def as_doc(d) -> str:
            if isinstance(d, int) and 0 <= d < len(ctx["doc_list"]):
                return ctx["doc_list"][d]
            if isinstance(d, (list, tuple)):
                return "\n".join(str(x) for x in _flatten(d))
            return str(d)

        def Retrieval(query=None, topk=5, **_):
            tick("Retrieval")
            try:
                k = max(1, min(int(topk), self.max_topk))
            except (TypeError, ValueError):
                k = 5
            q = as_query(query)
            hits = self.r.search(q, k)
            docs = [doc_contents(h) for h in hits]
            ctx["retrieved"] += len(docs)
            ctx["queries"].append(q)
            for h in hits:                       # para a análise de falhas: o que a execução chegou a buscar
                if h["id"] not in ctx["ret_ids"]:
                    ctx["ret_ids"].append(h["id"])
                    ctx["ret_titles"].append(h["title"])
            return docs

        def RewriteQuery(query=None, instruction="clarify", **_):
            tick("RewriteQuery")
            q = as_query(query)
            ins = str(instruction or "clarify").strip().lower()
            if ins == "clarify":
                pr, task = P.REWRITE_CLARIFY, "rc_rewrite_clarify"
                system = pr["system"]
            elif ins == "expand":
                pr, task = P.REWRITE_EXPAND, "rc_rewrite_expand"
                system = pr["system"]
            else:
                pr, task = P.REWRITE_CUSTOM, "rc_rewrite_custom"
                system = pr["system"].format(instruction=instruction)
            out = llm.chat(model, [{"role": "system", "content": system},
                                   {"role": "user", "content": pr["user"].format(query=q)}], task=task, max_tokens=256)
            obj = parse_obj(out, None)
            if isinstance(obj, dict):
                obj = obj.get("query", obj.get("queries", list(obj.values())))
            qs = as_str_list(obj)
            return [x.strip() for x in qs if x.strip()] or [q]

        def DecomposeQuery(query=None, **_):
            tick("DecomposeQuery")
            q = as_query(query)
            out = llm.chat(model, [{"role": "system", "content": P.DECOMPOSE["system"]},
                                   {"role": "user", "content": P.DECOMPOSE["user"].format(query=q)}],
                           task="rc_decompose", max_tokens=256)
            qs = as_str_list(parse_obj(out, None))
            return [x.strip() for x in qs if x.strip()][:5] or [q]

        def RefineDoc(query=None, doc="", instruction="summarize", **_):
            tick("RefineDoc")
            q, d = as_query(query), as_doc(doc)
            ins = str(instruction or "").strip().lower()
            if ins not in ("explain", "summarize"):
                return d
            pr = P.REFINE_EXPLAIN if ins == "explain" else P.REFINE_SUMMARIZE
            out = llm.chat(model, [{"role": "system", "content": pr["system"]},
                                   {"role": "user", "content": pr["user"].format(document=d, question=q)}],
                           task=f"rc_refine_{ins}", max_tokens=384)
            obj = parse_obj(out, None)
            if isinstance(obj, dict):
                for key in ("refined_document", "refined_content", "explanation"):
                    if obj.get(key):
                        return str(obj[key])
            return d

        def GenerateAnswer(query=None, docs=None, additional_instruction=None, **_):
            tick("GenerateAnswer")
            q = as_query(query)
            dl = [str(x) for x in _flatten(docs if docs is not None else ctx["doc_list"])][: self.max_docs]
            ctx["final_docs"] = dl
            reference = "".join(f"Passage: {d}\n" for d in dl)
            msgs = []
            if additional_instruction:
                msgs.append({"role": "system", "content": P.GENERATE_ANSWER_TIPS.format(
                    additional_instruction=str(additional_instruction))})
            msgs.append({"role": "user", "content": P.GENERATE_ANSWER_USER.format(reference=reference, question=q)})
            out = llm.chat(model, msgs, task="rc_generate_answer", max_tokens=384)
            ctx["last_answer"] = out
            return out

        fns = {"Retrieval": Retrieval, "RewriteQuery": RewriteQuery, "DecomposeQuery": DecomposeQuery,
               "RefineDoc": RefineDoc, "GenerateAnswer": GenerateAnswer}
        return {alias: fns[target] for alias, target in ALIASES.items()}

    # ------------------------------------------------------------------ execução
    def run(self, code: str, question: str, doc_list: list[str], previous_pred: str) -> dict:
        ctx = {"question": question, "doc_list": list(doc_list), "calls": [], "retrieved": 0,
               "final_docs": None, "last_answer": None, "queries": [], "ret_ids": [], "ret_titles": []}
        info = {"ok": False, "error": None, "calls": ctx["calls"], "answer_raw": ""}
        try:
            code = clean_code(code)
            tree = guard_loops(validate(code))
            ns = {"__builtins__": SAFE_BUILTINS, "__guard__": _guard, "question": question,
                  "doc_list": list(doc_list), "previous_pred": previous_pred, **self._mk(ctx)}
            exec(compile(tree, "<planner>", "exec"), ns)  # noqa: S102 — código validado pela whitelist acima
            var = final_answer_var(code)
            ans = ns.get(var)
            if not isinstance(ans, str):
                ans = ns.get("final_answer") if isinstance(ns.get("final_answer"), str) else ctx["last_answer"]
            if not isinstance(ans, str) or not ans.strip():
                raise ProgramError("o programa não produziu resposta (faltou final_answer = GenerateAnswer(...))")
            info.update(ok=True, answer_raw=ans)
        except BudgetExceeded as e:
            info["error"] = f"orçamento: {e}"
            if ctx["last_answer"]:
                info.update(ok=True, answer_raw=ctx["last_answer"], partial=True)
        except Exception as e:  # noqa: BLE001 — qualquer erro do programa gerado
            info["error"] = f"{type(e).__name__}: {e}"
        info["final_docs"] = ctx["final_docs"]
        info["n_calls"] = len(ctx["calls"])
        info["retrieved"] = ctx["retrieved"]
        info["queries"], info["ret_ids"], info["ret_titles"] = ctx["queries"], ctx["ret_ids"], ctx["ret_titles"]
        return info
