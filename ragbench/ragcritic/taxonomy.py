"""Sistema hierárquico de erros do RAG-Critic (7 → 19 → 4000+) e as Tabelas 11 e 12 do artigo.

Nomes de nível 1: o artigo (Fig. 2, Tabela 11) usa "Incomplete Response", "Inaccurate Response" e
"Off-Topic Response"; o benchmark e o script oficial de avaliação usam "Incomplete or Missing Response",
"Inaccurate or Misunderstood Response" e "Irrelevant or Off-Topic Response". ``canonical_tier1``
normaliza as duas grafias.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------- níveis 1 e 2 (script oficial)
TIER1 = [
    "Incomplete Information",
    "Irrelevant Information",
    "Erroneous Information",
    "Incomplete or Missing Response",
    "Inaccurate or Misunderstood Response",
    "Irrelevant or Off-Topic Response",
    "Overly Verbose Response",
]

TIER2 = [
    "Insufficient or Incomplete Information Retrieval",
    "Data Insufficiency in Retrieval",
    "Relevance Gaps in Retrieval",
    "Irrelevant Information Retrieval",
    "Erroneous Information Retrieval",
    "Omission of Key Information",
    "Lack of Specificity",
    "Specificity and Precision Errors",
    "Partial Coverage and Temporal Issues",
    "Lack of Practicality",
    "Contextual Understanding Errors",
    "Factual Inaccuracies",
    "Incorrect and Incomplete Answers",
    "Misinterpretation of Queries and Information",
    "Entity and Concept Confusion",
    "Irrelevant Content and Topic Drift",
    "Off-Topic and Redundant Responses",
    "Content and Context Misalignment",
    "Overly Complex and Redundant Response",
]

# pai de nível 1 de cada rótulo de nível 2 (all_tags_structure_final.json do repositório oficial)
TIER2_PARENT = {
    "Insufficient or Incomplete Information Retrieval": "Incomplete Information",
    "Data Insufficiency in Retrieval": "Incomplete Information",
    "Relevance Gaps in Retrieval": "Irrelevant Information",
    "Irrelevant Information Retrieval": "Irrelevant Information",
    "Erroneous Information Retrieval": "Erroneous Information",
    "Omission of Key Information": "Incomplete or Missing Response",
    "Lack of Specificity": "Incomplete or Missing Response",
    "Specificity and Precision Errors": "Incomplete or Missing Response",
    "Partial Coverage and Temporal Issues": "Incomplete or Missing Response",
    "Lack of Practicality": "Incomplete or Missing Response",
    "Contextual Understanding Errors": "Inaccurate or Misunderstood Response",
    "Factual Inaccuracies": "Inaccurate or Misunderstood Response",
    "Incorrect and Incomplete Answers": "Inaccurate or Misunderstood Response",
    "Misinterpretation of Queries and Information": "Inaccurate or Misunderstood Response",
    "Entity and Concept Confusion": "Inaccurate or Misunderstood Response",
    "Irrelevant Content and Topic Drift": "Irrelevant or Off-Topic Response",
    "Off-Topic and Redundant Responses": "Irrelevant or Off-Topic Response",
    "Content and Context Misalignment": "Irrelevant or Off-Topic Response",
    "Overly Complex and Redundant Response": "Overly Verbose Response",
}

_ALIASES = {
    "incomplete response": "Incomplete or Missing Response",
    "incomplete or missing response": "Incomplete or Missing Response",
    "inaccurate response": "Inaccurate or Misunderstood Response",
    "inaccurate or misunderstood response": "Inaccurate or Misunderstood Response",
    "off-topic response": "Irrelevant or Off-Topic Response",
    "irrelevant or off-topic response": "Irrelevant or Off-Topic Response",
    "overly verbose response": "Overly Verbose Response",
    "incomplete information": "Incomplete Information",
    "irrelevant information": "Irrelevant Information",
    "erroneous information": "Erroneous Information",
}

# retrieval (41,3% dos erros no artigo) × geração (58,7%)
STAGE = {t: ("retrieval" if i < 3 else "generation") for i, t in enumerate(TIER1)}


def canonical_tier1(tag: str) -> str | None:
    return _ALIASES.get((tag or "").strip().lower())


# ----------------------------------------------------------------------------- Tabela 11 (erro → ações)
ERROR_ACTION_TABLE = {
    "Incomplete Information": ["Re-retrieval: Rewrite the query for supplementary retrieval.",
                               "Rewriting the input: Refine the retrieved knowledge."],
    "Irrelevant Information": ["Re-retrieval: Perform replacement retrieval using the same query.",
                               "Rewriting the input: Correct the retrieved knowledge."],
    "Erroneous Information": ["Rewriting the input: Correct the retrieved knowledge.",
                              "Rewriting the reasoning answer: Correct the reasoning part."],
    "Incomplete or Missing Response": ["Re-retrieval: Perform supplementary retrieval using the same query.",
                                       "Rewriting the input: Provide examples for the retrieved knowledge."],
    "Inaccurate or Misunderstood Response": ["Rewriting the reasoning answer: Refine the reasoning part.",
                                             "Rewriting the input: Explain the retrieved knowledge."],
    "Irrelevant or Off-Topic Response": ["Re-retrieval: Rewrite the query for replacement retrieval.",
                                         "Rewriting the query: Break down the query into sub-questions."],
    "Overly Verbose Response": ["Rewriting the input: Refine the retrieved knowledge.",
                                "Rewriting the reasoning answer: Do not rely on the original reasoning part."],
}

# ----------------------------------------------------------------------------- Tabela 12 (5 funções, 15 sub-ações)
ACTION_FUNCTIONS = {
    "Retrieval": ["Perform supplementary retrieval using the same query.",
                  "Perform replacement retrieval using the same query.",
                  "Rewrite the query for supplementary retrieval.",
                  "Rewrite the query for replacement retrieval."],
    "Rewrite": ["Expand the query.", "Refine the query.", "Summarize the query.", "Clarify or explain the query."],
    "Decompose": ["Break down the query into sub-questions."],
    "Refine": ["Explain the retrieved knowledge.", "Refine the retrieved knowledge.", "Correct the retrieved knowledge.",
               "Delete the specific retrieved knowledge.", "Provide examples for the retrieved knowledge.",
               "Summarize the retrieved knowledge."],
    "Generate": ["Generate the final answer."],
}


def mapping_table_text(tier1_tags: list[str] | None = None) -> str:
    """Texto da Tabela 11 para o planner (só as linhas dos erros apontados, ou a tabela inteira)."""
    wanted = [canonical_tier1(t) for t in (tier1_tags or [])]
    wanted = [w for w in wanted if w]
    rows = wanted or list(ERROR_ACTION_TABLE)
    seen, out = set(), []
    for t in rows:
        if t in seen:
            continue
        seen.add(t)
        out.append(f"- {t}: " + " | ".join(ERROR_ACTION_TABLE[t]))
    return "\n".join(out)


# ----------------------------------------------------------------------------- programas por regra (ablação "w/o Auto-Planning")
RULE_PROGRAMS = {
    "Incomplete Information": (
        'new_queries = RewriteQuery(query=question, instruction="expand")\n'
        'new_docs = Retrieval(query=new_queries[0], topk=3)\n'
        'refined = [RefineDoc(query=question, doc=d, instruction="summarize") for d in new_docs]\n'
        'final_answer = GenerateAnswer(query=question, docs=doc_list + refined)'),
    "Irrelevant Information": (
        'replacement = Retrieval(query=question, topk=10)[5:]\n'
        'refined = [RefineDoc(query=question, doc=d, instruction="summarize") for d in replacement[:3]]\n'
        'final_answer = GenerateAnswer(query=question, docs=refined + replacement[3:])'),
    "Erroneous Information": (
        'explained = [RefineDoc(query=question, doc=d, instruction="explain") for d in doc_list[:3]]\n'
        'final_answer = GenerateAnswer(query=question, docs=explained, additional_instruction="The previous answer '
        '(" + previous_pred + ") relied on erroneous information. Correct the reasoning using only facts stated in the documents.")'),
    "Incomplete or Missing Response": (
        'more_docs = Retrieval(query=question, topk=8)\n'
        'final_answer = GenerateAnswer(query=question, docs=more_docs, additional_instruction="The previous answer '
        'was incomplete. Give a complete and specific answer.")'),
    "Inaccurate or Misunderstood Response": (
        'explained = [RefineDoc(query=question, doc=d, instruction="explain") for d in doc_list[:3]]\n'
        'final_answer = GenerateAnswer(query=question, docs=explained + doc_list[3:], additional_instruction="The previous '
        'answer (" + previous_pred + ") was inaccurate. Refine the reasoning step by step.")'),
    "Irrelevant or Off-Topic Response": (
        'sub_queries = DecomposeQuery(query=question)\n'
        'docs = []\n'
        'for sq in sub_queries[:3]:\n'
        '    docs = docs + Retrieval(query=sq, topk=3)\n'
        'final_answer = GenerateAnswer(query=question, docs=docs, additional_instruction="Answer exactly what the question asks.")'),
    "Overly Verbose Response": (
        'refined = [RefineDoc(query=question, doc=d, instruction="summarize") for d in doc_list[:3]]\n'
        'final_answer = GenerateAnswer(query=question, docs=refined, additional_instruction="Answer as briefly as possible. '
        'Do not rely on the previous reasoning.")'),
}


def rule_program(tier1_tags: list[str]) -> tuple[str, str]:
    for t in tier1_tags or []:
        c = canonical_tier1(t)
        if c in RULE_PROGRAMS:
            return c, RULE_PROGRAMS[c]
    return "Incomplete Information", RULE_PROGRAMS["Incomplete Information"]
