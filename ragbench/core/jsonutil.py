"""Parse tolerante de JSON vindo de LLMs pequenos (JSON quebrado, dict Python com aspas simples, cercas ```)."""
from __future__ import annotations

import ast
import json
import re

try:
    import json_repair
except ImportError:  # pragma: no cover
    json_repair = None

_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.S)


def parse_obj(text: str, default=None):
    """Devolve o primeiro objeto/lista JSON (ou literal Python) encontrado no texto."""
    if text is None:
        return default
    s = text.strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(s)
        except Exception:  # noqa: BLE001
            pass
    # recorta do primeiro { ou [ até o último } ou ]
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if starts:
        st = min(starts)
        end = max(s.rfind("}"), s.rfind("]"))
        if end > st:
            frag = s[st:end + 1]
            for loader in (json.loads, ast.literal_eval):
                try:
                    return loader(frag)
                except Exception:  # noqa: BLE001
                    pass
    if json_repair is not None:
        try:
            out = json_repair.loads(s)
            if out not in ("", None):
                return out
        except Exception:  # noqa: BLE001
            pass
    return default


def as_str_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, dict):
        for k in ("queries", "query", "subquestions", "sub_questions", "items", "list"):
            if k in v:
                return as_str_list(v[k])
        return [str(x) for x in v.values() if isinstance(x, str)]
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                out.extend(as_str_list(x))
            elif x is not None:
                out.append(str(x))
        return [x for x in out if x.strip()]
    return [str(v)]
