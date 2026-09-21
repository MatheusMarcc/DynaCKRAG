"""Prompts. Curtos de propósito: modelos de 3B seguem instruções curtas bem melhor.

O prompt do crítico segue o formato de entrada com que o RAG-Critic-3B foi treinado
(repositório RUC-NLPIR/RAG-Critic, licença MIT — ver THIRD_PARTY_NOTICES.md). Mudar esse
formato piora o crítico, porque ele é um modelo ajustado para exatamente essa entrada.
"""

# ------------------------------------------------------------------ laço A
ANSWER_SYSTEM = (
    "Answer the question using only the given evidence. The answer must be as short as possible: "
    "an entity, a date, a number, or yes/no. End with the line 'So the final answer is: <answer>'."
)
ANSWER_USER = "Evidence:\n{evidence}\n\nQuestion: {question}"
ANSWER_TIPS = "\nPay attention to this feedback about a previous wrong attempt: {tips}"

SUFFICIENCY_SYSTEM = (
    "You decide whether the passages contain ALL facts needed to answer a multi-hop question. "
    "If something is missing, describe the single most important missing fact as a short search query."
)
SUFFICIENCY_USER = "Question: {question}\n\nPassages:\n{passages}"
SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {"sufficient": {"type": "boolean"}, "missing": {"type": "string"}},
    "required": ["sufficient", "missing"],
}

REWRITE_SYSTEM = (
    "Write ONE new search query that would retrieve the evidence still needed to answer the question. "
    "Reuse names and facts found in the passages. Output only the query."
)
REWRITE_USER = "Question: {question}\nPrevious query: {query}\n\nPassages:\n{passages}\n\nNew query:"

COMPRESS_SYSTEM = (
    "Extract at most {n} short snippets from the passages that together support answering the question. "
    "Keep names, dates and numbers exactly as written. One snippet per line, starting with '- '."
)
COMPRESS_USER = "Question: {question}\n\nPassages:\n{passages}\n\nSnippets:"

# ------------------------------------------------------------------ crítico (formato RAG-Critic)
_CRITIC_HEAD = (
    "You are a critical system designed to provide useful error type tags for retrieval-augmented "
    "generation (RAG) tasks. Your goal is to assist in detailed error analysis to improve the performance "
    "of AI assistants. Below are the [Question], the top-5 retrieved relevant [Passages], and the "
    "[Model's Prediction] for the RAG tasks.\n\n"
)
_IND = " " * 20
_CRITIC_TAIL = (
    "\n\n" + _IND + "Please first determine whether the model's prediction is correct. "
    "If it is correct, output it as follows:\n" + _IND + '{"Judgement": "Correct"}\n\n' + _IND
    + "If it is incorrect, please identify the error tags at three levels, from coarse to fine, and provide "
    "a detailed error analysis. Adhere strictly to the following JSON format:\n"
    + _IND + "{\n" + _IND + '    "Judgement": "Error",\n' + _IND + '    "Error_analysis": "",\n'
    + _IND + '    "tag1": [],\n' + _IND + '    "tag2": [],\n' + _IND + '    "tag3": []\n' + _IND + "}\n" + _IND
)


def critic_prompt(question: str, passages: list[str], prediction: str) -> str:
    body = _IND + f"Question: {question}\n"
    for i, p in enumerate(passages, 1):
        body += _IND + f"Passage {i}: {p}\n"
    body += _IND + f"Model's Prediction: {prediction}"
    return _CRITIC_HEAD + body + _CRITIC_TAIL


# ------------------------------------------------------------------ laço B (reparo)
# Tabela erro→ação (Tabela 11 do artigo RAG-Critic), indexada por palavras-chave do tag1
ERROR_ACTION_TABLE = {
    "incomplete information": "retrieve more with a rewritten query; summarize the documents",
    "irrelevant": "replace the documents using a new retrieval with the same query; drop irrelevant documents",
    "erroneous": "correct the documents; do not trust the previous reasoning",
    "incomplete response": "retrieve more with the same query; give a complete answer",
    "inaccurate": "re-read the evidence carefully and fix the reasoning",
    "off-topic": "decompose the question into sub-questions and retrieve for each",
    "verbose": "answer with only the short final answer",
}


def error_hints(tag1: list[str]) -> str:
    hints = []
    low = " | ".join(t.lower() for t in tag1)
    for key, action in ERROR_ACTION_TABLE.items():
        if key in low and action not in hints:
            hints.append(f"- {key}: {action}")
    return "\n".join(hints) or "- (no mapped hint)"


PLANNER_CODE_SYSTEM = """You fix a failed retrieval-augmented answer by writing short Python code.
Available functions (nothing else is allowed, no imports, no loops except list comprehensions):
- Retrieval(query: str, topk: int) -> list[str]            # search the corpus
- RewriteQuery(query: str, instruction: str) -> list[str]  # instruction: "clarify" or "expand"
- DecomposeQuery(query: str) -> list[str]                  # sub-questions
- RefineDoc(query: str, doc: str, instruction: str) -> str # instruction: "summarize" or "explain"
- GenerateAnswer(query: str, docs: list[str], additional_instruction: str) -> str
Available variables: question, doc_list, previous_pred.
Call only what is necessary. The last line must be: final_answer = GenerateAnswer(...)
Reply with a single ```python code block."""

PLANNER_CODE_USER = """question = {question!r}
doc_list = <{n_docs} documents, titles: {titles}>
previous_pred = {pred!r}

Critic verdict: the previous prediction is WRONG.
Error types: {tags}
Critic analysis: {analysis}
Suggested fixes:
{hints}"""

PLANNER_MENU_SYSTEM = """You fix a failed retrieval-augmented answer by choosing repair steps.
Allowed actions:
- retrieve_same: retrieve more documents with the original question (argument: "")
- retrieve_new: retrieve with a better search query (argument: the new query)
- decompose: split the question into sub-questions and retrieve for each (argument: "")
- summarize_docs: shrink every document to the facts relevant to the question (argument: "")
- drop_docs: remove irrelevant documents (argument: comma-separated document numbers, e.g. "2,5")
Choose at most 3 steps, then write a short instruction for the final answer."""

PLANNER_MENU_USER = """Question: {question}
Documents:
{passages}
Previous prediction: {pred}
Critic verdict: WRONG. Error types: {tags}
Critic analysis: {analysis}
Suggested fixes:
{hints}"""

PLANNER_MENU_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array", "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["retrieve_same", "retrieve_new", "decompose",
                                                          "summarize_docs", "drop_docs"]},
                    "argument": {"type": "string"},
                },
                "required": ["action", "argument"],
            },
        },
        "answer_instruction": {"type": "string"},
    },
    "required": ["steps", "answer_instruction"],
}

DECOMPOSE_SYSTEM = "Split the question into 2 or 3 simple sub-questions that can be searched independently."
DECOMPOSE_SCHEMA = {"type": "object", "properties": {"subquestions": {"type": "array", "items": {"type": "string"}}},
                    "required": ["subquestions"]}

REFINE_SYSTEM = {
    "summarize": "Rewrite the document keeping only the facts that help answer the question. Output only the text.",
    "explain": "Explain what the document says that is relevant to the question. Output only the explanation.",
}
REFINE_USER = "Question: {question}\n\nDocument:\n{doc}"

GAP_FROM_CRITIQUE_SYSTEM = (
    "A critic explained why an answer was wrong. Write ONE short search query for the missing or "
    "wrong fact that must be found to answer correctly. Output only the query."
)
GAP_FROM_CRITIQUE_USER = "Question: {question}\nCritic analysis: {analysis}\n\nSearch query:"
