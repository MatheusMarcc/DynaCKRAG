"""Contagem de tokens independente do cache de KV do Ollama.

O Ollama reaproveita prefixos já processados e, nesses casos, ``prompt_eval_count`` conta só os
tokens NOVOS. Para comparar o custo dos métodos (coluna Tok. da Tabela 1 do DynaKRAG) precisamos
do total real de tokens de cada chamada, então contamos com o tokenizer do Qwen2.5
(baixado do HuggingFace na primeira vez). Sem ele, cai numa estimativa de ~4 caracteres por token.
A estimativa é a mesma para todos os métodos, então as comparações continuam justas.
"""
from __future__ import annotations

import threading

_TOK = None
_TRIED = False
_LOCK = threading.Lock()
TOKENIZER_REPO = "Qwen/Qwen2.5-3B-Instruct"


def _load():
    global _TOK, _TRIED
    with _LOCK:
        if _TRIED:
            return _TOK
        _TRIED = True
        try:
            from tokenizers import Tokenizer  # vem junto com transformers/sentence-transformers

            _TOK = Tokenizer.from_pretrained(TOKENIZER_REPO)
        except Exception:
            _TOK = None
        return _TOK


def disable_tokenizer() -> None:
    """Usado no modo smoke/testes: não tenta baixar nada."""
    global _TRIED, _TOK
    _TRIED, _TOK = True, None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tok = _load()
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False).ids)
        except Exception:
            pass
    return max(1, round(len(text) / 4))


def count_messages(messages: list[dict]) -> int:
    # ~4 tokens de moldura por mensagem no template ChatML (<|im_start|>role\n ... <|im_end|>\n)
    return sum(count_tokens(m.get("content", "")) + 4 for m in messages) + 3


def tokenizer_kind() -> str:
    return "qwen2.5" if _load() is not None else "heuristic(chars/4)"
