"""Prompts do DynaKRAG e dos baselines controlados.

O artigo não publica os prompts. Estes são curtos de propósito (modelos de 3B seguem instruções
curtas melhor) e são OS MESMOS para todos os métodos onde a etapa é a mesma: todos respondem com o
mesmo prompt de resposta (``ANSWER_*``), como exige o protocolo "same backbone, same evaluation code".
"""

# ----------------------------------------------------------------------------- gerador G (resposta)
ANSWER_SYSTEM = (
    "You are a question answering assistant. Answer the question using the evidence. "
    "Reply with the short answer only (an entity, a date, a number, or yes/no), with no explanation."
)
ANSWER_USER = "Evidence:\n{evidence}\n\nQuestion: {question}\nAnswer:"

NO_RETRIEVAL_SYSTEM = (
    "Answer the question from your own knowledge. Reply with the short answer only (an entity, a date, "
    "a number, or yes/no), with no explanation."
)
NO_RETRIEVAL_USER = "Question: {question}\nAnswer:"

# ----------------------------------------------------------------------------- sufficiency_check (sonda)
# SUFF_SYSTEM + GAP_SYSTEM são o modo two_step (duas chamadas), mantido para reproduzir resultados antigos.
SUFF_SYSTEM = (
    "You check whether the documents contain ALL the facts needed to answer the question "
    "(it may need several reasoning hops). Reply in JSON: {\"sufficient\": true} or {\"sufficient\": false}."
)
SUFF_USER = "Question: {question}\n\nDocuments:\n{passages}"
SUFF_SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}}, "required": ["sufficient"]}

GAP_SYSTEM = (
    "The documents are NOT enough to answer the question. Describe the single most important missing fact "
    "as a short search query (at most 12 words). Reply with the query only."
)
GAP_USER = "Question: {question}\n\nDocuments:\n{passages}\n\nMissing information:"

# modo one_step (o padrão): o julgamento e a lacuna numa chamada só, com os mesmos critérios dos dois prompts acima
SUFF_ONE_SYSTEM = (
    "You check whether the documents contain ALL the facts needed to answer the question "
    "(it may need several reasoning hops). If something is missing, describe the single most important missing "
    "fact as a short search query (at most 12 words). "
    "Reply in JSON with keys 'sufficient' (true/false) and 'missing' (the search query; empty string if sufficient)."
)
SUFF_ONE_SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}, "missing": {"type": "string"}},
                   "required": ["sufficient", "missing"]}

# ----------------------------------------------------------------------------- rewrite_query
REWRITE_SYSTEM = (
    "Write ONE new search query to retrieve the evidence that is still missing to answer the question. "
    "Use names and facts found in the documents (for example, an intermediate entity). Reply with the query only."
)
REWRITE_USER = "Question: {question}\nCurrent query: {query}\n\nDocuments:\n{passages}\n\nNew query:"

# ----------------------------------------------------------------------------- compress_answer_evidence
COMPRESS_EXTRACT_SYSTEM = (
    "Extract at most {n} short snippets from the documents that together support the answer to the question. "
    "Copy names, dates and numbers exactly. Write one snippet per line starting with '- '. "
    "Do not answer the question."
)
COMPRESS_TABLE_SYSTEM = (
    "Build a compact fact table with the facts from the documents that are needed to answer the question, "
    "following the chain of reasoning hop by hop. Write at most {n} rows, one per line, in the format: "
    "entity | relation | value. Copy names, dates and numbers exactly. Do not answer the question."
)
COMPRESS_USER = "Question: {question}\n\nDocuments:\n{passages}\n\nOutput:"

# ============================================================================= baselines
IRCOT_SYSTEM = (
    "You answer multi-hop questions by reasoning over the documents one step at a time. Write only the NEXT "
    "single sentence of reasoning, grounded in the documents. When the reasoning is complete, write: "
    "So the answer is: <short answer>.\n\n"
    "Example\nQuestion: In which country was the director of the film Night Train born?\n"
    "Reasoning so far: Night Train was directed by John Doe.\n"
    "Next sentence: John Doe was born in Toronto, Canada.\n"
    "Next sentence: So the answer is: Canada."
)
IRCOT_USER = "Documents:\n{passages}\n\nQuestion: {question}\nReasoning so far: {cot}\nNext sentence:"

SELF_ASK_SYSTEM = """Answer the question by asking and answering simpler follow-up questions, in this exact format.

Question: Who was born first, Albert Einstein or Nikola Tesla?
Are follow up questions needed here: Yes.
Follow up: When was Albert Einstein born?
Intermediate answer: March 14, 1879.
Follow up: When was Nikola Tesla born?
Intermediate answer: July 10, 1856.
So the final answer is: Nikola Tesla

Question: Where was the director of the film Jaws born?
Are follow up questions needed here: Yes.
Follow up: Who directed the film Jaws?
Intermediate answer: Steven Spielberg.
Follow up: Where was Steven Spielberg born?
Intermediate answer: Cincinnati, Ohio.
So the final answer is: Cincinnati, Ohio"""
SELF_ASK_USER = "Question: {question}\nAre follow up questions needed here: Yes.\n{transcript}"

CRAG_EVAL_SYSTEM = (
    "You are a retrieval evaluator. For each document, judge whether it contains information that is relevant "
    "and helpful for answering the question: 'correct' (clearly relevant), 'ambiguous' (partially relevant) or "
    "'incorrect' (irrelevant). Reply in JSON: {\"labels\": [...]} with one label per document, in order."
)
CRAG_EVAL_USER = "Question: {question}\n\nDocuments:\n{passages}"
CRAG_EVAL_SCHEMA = {"type": "object", "properties": {"labels": {"type": "array", "items": {
    "type": "string", "enum": ["correct", "ambiguous", "incorrect"]}}}, "required": ["labels"]}
CRAG_REFINE_SYSTEM = (
    "Below are knowledge strips taken from retrieved documents. Keep only the strips that are relevant to the "
    "question and copy them, one per line, starting with '- '. Keep at most {n} strips."
)
CRAG_REFINE_USER = "Question: {question}\n\nKnowledge strips:\n{strips}\n\nRelevant strips:"
CRAG_REWRITE_SYSTEM = "Rewrite the question as a short keyword search query (at most 10 words). Reply with the query only."
CRAG_REWRITE_USER = "Question: {question}\nSearch query:"

ADAPTIVE_SYSTEM = (
    "Classify the complexity of the question. A: can be answered without any retrieval. B: needs one retrieval "
    "step. C: needs multi-step reasoning with several retrievals. Reply in JSON: {\"label\": \"A\"|\"B\"|\"C\"}."
)
ADAPTIVE_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["A", "B", "C"]}},
                   "required": ["label"]}

S2G_SYSTEM = (
    "You judge whether the documents are sufficient to answer the question and, if not, identify the specific "
    "missing facts. Reply in JSON with keys: 'sufficient' (true/false), 'known_facts' (list of short facts "
    "found in the documents) and 'gaps' (list of objects with 'missing' — what is missing — and 'query' — a "
    "short search query to find it). Give at most 2 gaps."
)
S2G_USER = "Question: {question}\n\nDocuments:\n{passages}"
S2G_SCHEMA = {"type": "object", "properties": {
    "sufficient": {"type": "boolean"},
    "known_facts": {"type": "array", "items": {"type": "string"}},
    "gaps": {"type": "array", "items": {"type": "object", "properties": {
        "missing": {"type": "string"}, "query": {"type": "string"}}, "required": ["missing", "query"]}}},
    "required": ["sufficient", "known_facts", "gaps"]}

CORAG_SUBQ_SYSTEM = (
    "You answer a complex question by retrieving information step by step. Given the question and the "
    "sub-questions answered so far, write the NEXT short sub-question to search for. Reply with the sub-question only."
)
CORAG_SUBQ_USER = "Question: {question}\n\nSteps so far:\n{chain}\n\nNext sub-question:"

PAR2_DECOMPOSE_SYSTEM = (
    "Break the question into at most {n} simple sub-questions that together cover all the facts needed to answer "
    "it. Reply in JSON: {{\"subquestions\": [...]}}."
)
PAR2_DECOMPOSE_USER = "Question: {question}"
PAR2_DECOMPOSE_SCHEMA = {"type": "object", "properties": {"subquestions": {"type": "array", "items": {"type": "string"}}},
                         "required": ["subquestions"]}
