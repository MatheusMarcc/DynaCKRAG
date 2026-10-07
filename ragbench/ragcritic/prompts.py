"""Prompts do RAG-Critic.

* Prompt A (geração), B (análise de erro), C (tags open-set): texto do apêndice B.5 do artigo.
* Prompt D (crítico ajustado) e E (baselines no RAG-Error Bench): reconstruídos a partir dos arquivos
  oficiais ``critic_model_test.json`` e ``baseline_test.json`` — a renderização abaixo reproduz os 3.800
  prompts do benchmark **byte a byte** (verificado nos testes).
* Prompt F (planner) e os prompts das 5 funções: ``critic_agent/prompt_template.py`` e
  ``critic_agent/agent_prompt_template.py`` do repositório oficial (licença MIT, ver THIRD_PARTY_NOTICES.md).

ATENÇÃO — artefato de formato no benchmark oficial: nos dois arquivos, TODOS os exemplos "Correct" usam
recuo de 20 espaços e TODOS os exemplos "Error" usam recuo de 16 espaços. O recuo sozinho separa as classes
com 100% de acerto. Por isso o crítico é avaliado também com o recuo trocado (auditoria de formato em
``errorbench.py``), e o recuo usado no fluxo agêntico é configurável (``ragcritic.critic_template``).
"""
from __future__ import annotations

HEAD = "You are a critical system designed to provide useful error type tags for retrieval-augmented generation (RAG) tasks. Your goal is to assist in detailed error analysis to improve the performance of AI assistants. Below are the [Question], the top-5 retrieved relevant [Passages], and the [Model's Prediction] for the RAG tasks.\n\n"
CRITIC_TAIL_20 = '\n                    Please first determine whether the model\'s prediction is correct. If it is correct, output it as follows:\n                    {"Judgement": "Correct"}\n\n                    If it is incorrect, please identify the error tags at three levels, from coarse to fine, and provide a detailed error analysis. Adhere strictly to the following JSON format:\n                    {\n                        "Judgement": "Error",\n                        "Error_analysis": "",\n                        "tag1": [],\n                        "tag2": [],\n                        "tag3": []\n                    }\n                    '
CRITIC_TAIL_16 = '\n                Please first determine whether the model\'s prediction is correct. If it is correct, output it as follows:\n                {"Judgement": "Correct"}\n\n                If it is incorrect, please identify the error tags at three levels, from coarse to fine, and provide a detailed error analysis. Adhere strictly to the following JSON format:\n                {\n                    "Judgement": "Error",\n                    "Error_analysis": "",\n                    "tag1": [],\n                    "tag2": [],\n                    "tag3": []\n                }\n                '
BASELINE_TAIL_20 = '\n                    Please first determine whether the model\'s prediction is correct. If it is correct, output it as follows:\n                    {"Judgement": "Correct"}\n\n                    If it is incorrect, give these error types, tag1 corresponds to tag2 one-to-one：\n                            tag1= [\n                                    "Incomplete Information",\n                                    "Incomplete Information",\n                                    "Irrelevant Information",\n                                    "Irrelevant Information",\n                                    "Erroneous Information",\n                                    "Incomplete or Missing Response",\n                                    "Incomplete or Missing Response",\n                                    "Incomplete or Missing Response",\n                                    "Incomplete or Missing Response",\n                                    "Incomplete or Missing Response",\n                                    "Inaccurate or Misunderstood Response",\n                                    "Inaccurate or Misunderstood Response",\n                                    "Inaccurate or Misunderstood Response",\n                                    "Inaccurate or Misunderstood Response",\n                                    "Inaccurate or Misunderstood Response",\n                                    "Inaccurate or Misunderstood Response",\n                                    "Irrelevant or Off-Topic Response",\n                                    "Irrelevant or Off-Topic Response",\n                                    "Irrelevant or Off-Topic Response",\n                                    "Overly Verbose Response"\n                                ]\n\n                                tag2 = [\n                                    "Insufficient or Incomplete Information Retrieval",\n                                    "Data Insufficiency in Retrieval",\n                                    "Relevance Gaps in Retrieval",\n                                    "Irrelevant Information Retrieval",\n                                    "Erroneous Information Retrieval",\n                                    "Omission of Key Information",\n                                    "Lack of Specificity",\n                                    "Specificity and Precision Errors",\n                                    "Partial Coverage and Temporal Issues",\n                                    "Lack of Practicality",\n                                    "Contextual Understanding Errors",\n                                    "Factual Inaccuracies",\n                                    "Incorrect and Incomplete Answers",\n                                    "Golden Answer Misalignment",\n                                    "Misinterpretation of Queries and Information",\n                                    "Entity and Concept Confusion",\n                                    "Irrelevant Content and Topic Drift",\n                                    "Off-Topic and Redundant Responses",\n                                    "Content and Context Misalignment",\n                                    "Overly Complex and Redundant Response"\n                                ]\n                    Please identify the error tags at three levels, from coarse to fine, and provide a detailed error analysis. Adhere strictly to the following JSON format:\n                    {\n                        "Judgement": "Error",\n                        "Error_analysis": "",\n                        "tag1": [],\n                        "tag2": [],\n                        "tag3": []\n                    }\n                '
BASELINE_TAIL_16 = '\n                Please first determine whether the model\'s prediction is correct. If it is correct, output it as follows:\n                {"Judgement": "Correct"}\n\n                If it is incorrect, give these error types, tag1 corresponds to tag2 one-to-one：\n                        tag1= [\n                                "Incomplete Information",\n                                "Incomplete Information",\n                                "Irrelevant Information",\n                                "Irrelevant Information",\n                                "Erroneous Information",\n                                "Incomplete or Missing Response",\n                                "Incomplete or Missing Response",\n                                "Incomplete or Missing Response",\n                                "Incomplete or Missing Response",\n                                "Incomplete or Missing Response",\n                                "Inaccurate or Misunderstood Response",\n                                "Inaccurate or Misunderstood Response",\n                                "Inaccurate or Misunderstood Response",\n                                "Inaccurate or Misunderstood Response",\n                                "Inaccurate or Misunderstood Response",\n                                "Inaccurate or Misunderstood Response",\n                                "Irrelevant or Off-Topic Response",\n                                "Irrelevant or Off-Topic Response",\n                                "Irrelevant or Off-Topic Response",\n                                "Overly Verbose Response"\n                            ]\n\n                            tag2 = [\n                                "Insufficient or Incomplete Information Retrieval",\n                                "Data Insufficiency in Retrieval",\n                                "Relevance Gaps in Retrieval",\n                                "Irrelevant Information Retrieval",\n                                "Erroneous Information Retrieval",\n                                "Omission of Key Information",\n                                "Lack of Specificity",\n                                "Specificity and Precision Errors",\n                                "Partial Coverage and Temporal Issues",\n                                "Lack of Practicality",\n                                "Contextual Understanding Errors",\n                                "Factual Inaccuracies",\n                                "Incorrect and Incomplete Answers",\n                                "Golden Answer Misalignment",\n                                "Misinterpretation of Queries and Information",\n                                "Entity and Concept Confusion",\n                                "Irrelevant Content and Topic Drift",\n                                "Off-Topic and Redundant Responses",\n                                "Content and Context Misalignment",\n                                "Overly Complex and Redundant Response"\n                            ]\n                 Please identify the error tags at three levels, from coarse to fine, and provide a detailed error analysis. Adhere strictly to the following JSON format:\n                {\n                    "Judgement": "Error",\n                    "Error_analysis": "",\n                    "tag1": [],\n                    "tag2": [],\n                    "tag3": []\n                }\n                '


# ============================================================================= Prompt A — geração (π_α)
PROMPT_A = (
    "Find the useful content from the provided documents, then answer the question. Answer the question directly. "
    "Your response should be very concise. Please provide the final answer is:' as a prefix for the final answer. "
    "The following are the given documents.\n\n{passages}\n\nAnswer the question directly. Your response should be "
    "very concise. Please use 'So the final answer is:' as a prefix for the final answer. **Question**: {question} "
    "**Response**:"
)


def passages_block(contents: list[str]) -> str:
    """'Passage: <conteúdo>' por documento (como no GenerateAnswer oficial)."""
    return "\n".join(f"Passage: {c}" for c in contents)


# ============================================================================= Prompt B — análise detalhada do erro (etapa 1)
PROMPT_B = (
    "You are an expert in error analysis for retrieval-augmented generation tasks. We will provide you with a prompt "
    "that includes both the question and relevant knowledge, along with a model's prediction and the golden answer. "
    "The details are as follows:\n\nPrompt: {rag_input}\n\nModel's Prediction: {prediction}\n\nGolden Answer: {golden}\n\n"
    "If the model's prediction is incorrect, please respond with a single JSON including the judgement in key "
    "'Judgement' and a detailed error analysis in key 'Error analysis'. Here is an example of output JSON format:\n\n"
    "{{'Judgement': \"incorrect\", 'Error analysis': \"The model's prediction is incorrect because ...\"}}\n\n"
    "If the model's prediction is correct, please respond with a single JSON as follows:\n\n"
    "{{'Judgement': \"Correct\", 'Error analysis': \"None\"}}"
)

# ============================================================================= Prompt C — tags open-set (etapa 2)
PROMPT_C = (
    "You are a tagging system designed to provide useful error type tags for retrieval-augmented generation (RAG) "
    "tasks. Your goal is to assist in detailed error analysis to improve the performance of AI assistants. Below is a "
    "detailed error analysis:\n\n{analysis}\n\nPlease provide fine-grained error tags to identify the main error types "
    "in the analysis.\nYour response should be a list that includes the titles of the error tags along with a brief "
    "explanation for each tag. Please adhere strictly to the following JSON format:\n\n"
    "{{\"tag\": \"\", \"explanation\": \"\"}}\n\nPlease respond in English."
)


# ============================================================================= Prompts D e E — crítico
def render_critic(question: str, passages: list[str], prediction: str, indent: int = 20, kind: str = "critic") -> str:
    """kind='critic' → Prompt D (modelo ajustado); kind='baseline' → Prompt E (com as listas de tags)."""
    if indent not in (16, 20):
        raise ValueError("indent deve ser 16 ou 20 (os dois formatos do benchmark oficial)")
    ind = " " * indent
    s = HEAD + ind + "Question: " + question
    for i, p in enumerate(passages, 1):
        s += "\n" + ind + f"Passage {i}: " + p
    s += "\n" + ind + "Model's Prediction: " + prediction
    tails = {("critic", 20): CRITIC_TAIL_20, ("critic", 16): CRITIC_TAIL_16,
             ("baseline", 20): BASELINE_TAIL_20, ("baseline", 16): BASELINE_TAIL_16}
    return s + tails[(kind, indent)]


def parse_critic_instruction(ins: str) -> dict:
    """Inverso de ``render_critic``: separa pergunta, 5 passagens e predição de um prompt do benchmark."""
    import re

    m = re.search(r"\n\n( *)Question: ", ins)
    ind = m.group(1)
    q_start = m.end()
    p1 = ins.index("\n" + ind + "Passage 1: ", q_start)
    question = ins[q_start:p1]
    passages, pos, i = [], p1, 1
    while True:
        tag = "\n" + ind + f"Passage {i}: "
        if ins.find(tag, pos) != pos:
            break
        st = pos + len(tag)
        nxt = ins.find("\n" + ind + f"Passage {i + 1}: ", st)
        end_pred = ins.find("\n" + ind + "Model's Prediction: ", st)
        en = nxt if nxt != -1 and nxt < end_pred else end_pred
        passages.append(ins[st:en])
        pos, i = en, i + 1
    pr_tag = "\n" + ind + "Model's Prediction: "
    pst = ins.index(pr_tag, pos) + len(pr_tag)
    ten = ins.index("\n" + ind + "Please first determine", pst)
    kind = "baseline" if "tag1 corresponds to tag2" in ins[ten:] else "critic"
    return {"question": question, "passages": passages, "prediction": ins[pst:ten], "indent": len(ind), "kind": kind}


# ============================================================================= Prompt F — planner (π_β), texto oficial
PLANNER_SYSTEM = """
You are an agent tasked with optimizing a Retrieval-Augmented Generation process. The goal is to improve the model's predictions by addressing issues flagged in the error_type. You are given the results from an initial RAG process, including a query, a list of retrieved documents, a prediction, and the identified error type. Your task is to optimize the current RAG process by selecting the appropriate functions and generating the corresponding Python code to fix the problem.

### Available Functions

1. **`Retrieval(query: str, topk: int) -> List[str]`**  
   **Purpose**: Retrieves the top-k most relevant documents for a given query from the corpus.  
   **Parameters**:
   - `query` (`str`): The input query string to retrieve relevant documents.  
   - `topk` (`int`): The number of top documents to return.  
   **Returns**:
   - A list of `topk` relevant document strings, sorted by relevance.

2. **`RewriteQuery(query: str, instruction: str) -> List[str]`**  
   **Purpose**: Rewrites the query based on the provided instruction to better match relevant documents.  
   **Parameters**:
   - `query` (`str`): The original query string to be rewritten.  
   - `instruction` (`str`): The instruction for rewriting the query. Possible instructions include:
     - `"clarify"`: Make the query more specific.
     - `"expand"`: Add more context or related terms to the query.  
   **Returns**:
   - A list of rewritten query strings, each representing a possible version of the query.

3. **`DecomposeQuery(query: str) -> List[str]`**  
   **Purpose**: Breaks down the input query into smaller, more specific sub-queries.  
   **Parameters**:
   - `query` (`str`): The original query string to decompose.  
   **Returns**:
   - A list of sub-query strings, which represent different aspects or more specific details of the original query.

4. **`RefineDoc(query: str, doc: str, instruction: str) -> str`**  
   **Purpose**: Refine a document in the doc_list (index starts from 0) based on the query. Use this function when you find some document in the doc_list is not relevant to the question.  
   **Parameters**:
   - `query` (`str`): The input query string.  
   - `doc` (`str`): The document need to refine.  
   - `instruction` (`str`): The instruction for refining the document. Supported instructions include:
     - `"explain"`: Provide a detailed explanation of the document.
     - `"summarize"`: Summarize the document.  
   **Returns**:
   - The refined document as a string, which could be either an explanation or a summary.

5. **`GenerateAnswer(query: str, docs: List[str], additional_instruction: str = None) -> str`**  
   **Purpose**: Generates an answer based on the query and relevant documents, incorporating additional instructions for answer improvement.  
   **Parameters**:
   - `query` (`str`): The input query string.  
   - `docs` (`List[str]`): A list of relevant documents used to generate the answer.  
   - `additional_instruction` (`str`): Additional instruction describing issues in the previous answer and desired improvements (e.g., requirements for precision, conciseness, or additional information).  
   **Returns**:
   - A generated answer string, potentially incorporating information from the documents, adjusted according to the provided instruction.
You can directly use the variables I provide to act as the input of the functions. You can freely combine the functions to improve the performance.
"""

PLANNER_USER = """Given the following information:

question = "{question}"
doc_list = {doc_list}
previous_pred = "{previous_pred}"

Error type of previous pred: {error_type}
{extra}
Please carefully read the `provided question`, `doc list`, `previous answer` and the error type of previous pred given by a teacher model. **Your task is to generate the Python code that calls the relevant functions to optimize the current RAG process and solve the previous error**. The generated code should only include function calls (do not write the function implementations). The order of the function calls should be appropriate to address the error type and improve the result. To save costs, you need to ensure that **each function execution is necessary** and can bring about an improvement in performance**. Only give the code, do not give any other explanation. You must use `final_answer =GenerateAnswer(...)` in the final to generate the final answer.
"""

PLANNER_EXTRA_PAPER = """
Coarse-grained error types (tier 1): {tier1}
Critic's error analysis: {analysis}

Error-action mapping table (reference solutions for these error types):
{table}
"""

# ============================================================================= prompts das 5 funções (oficiais)
REWRITE_CLARIFY = {
    "system": """
Please rewrite the given query to make it more specific, clear, and focused. Ensure that the revised query addresses the topic directly and eliminates any ambiguity. If necessary, add details or context to make the query more precise, so it can be better understood and answered. Output only the revised query in JSON format, with the query as the value of the 'query' key.

**Example Output (in JSON format):**
```json
{
  "query": "Your rewritten query goes here."
}
```""",
    "user": "Original query: {query}",
}
REWRITE_EXPAND = {
    "system": """
Please rewrite the given query by expanding it with additional relevant questions or variations that address the same topic. These expanded queries should help to explore different aspects or approaches related to the original query. You can introduce new perspectives, specific sub-questions, or related queries. Ensure the expanded queries are still coherent and closely related to the original question. Output only a JSON array, where each element is an expanded query as a string.

**Example Output (in JSON format):**
```json
[
  "Expanded query 1 goes here.",
  "Expanded query 2 goes here.",
  "Expanded query 3 goes here."
]
```""",
    "user": "Original query: {query}",
}
REWRITE_CUSTOM = {
    "system": """
Please rewrite the given query based on the following instruction: {instruction}. The rewritten query should align with the given directive and ensure clarity, specificity, and relevance to the original topic. Make sure the revised query is well-structured and can be easily understood, while following the instruction provided.

**Example Output (in JSON format):**
```json
{{
  "query": "Your rewritten query goes here."
}}
```""",
    "user": "Original query: {query}",
}
DECOMPOSE = {
    "system": """
Please split the given query into multiple smaller, more specific subqueries. Each subquery should focus on a different aspect of the original query and be concise and clear. The subqueries should be independent, but together they should still address the overall topic of the original query. Ensure that each subquery is well-formed, focused, and easy to understand. Output only a JSON array, where each element is a subquery as a string.

**Example Output (in JSON format):**
```json
[
  "Subquery 1 goes here.",
  "Subquery 2 goes here.",
  "Subquery 3 goes here."
]
```""",
    "user": "Original query: {query}",
}
REFINE_SUMMARIZE = {
    "system": """
Please refine the given document to retain only the information helpful for answering the provided question. Remove any details or content that is irrelevant to the query. Ensure the refined document focuses on the key points that answer the query and is concise, clear, and easy to understand. The refined content should reflect the most relevant information while discarding any unnecessary or unrelated sections. Output only the refined content in JSON format, with the refined content as the value of the 'refined_content' key.

**Example Output (in JSON format):**
```json
{
  "refined_document": "Your refined content related to the question goes here."
}
```""",
    "user": "Document: {document}\nQuestion: {question}",
}
REFINE_EXPLAIN = {
    "system": """
Please read the given document carefully and think critically about it in relation to the provided question. Your task is to provide a detailed explanation that answers the question, using the information from the document. Your explanation should not only focus on directly answering the question but should also provide reasoning, analysis, and any insights that emerge from the document. Make sure your response is logical, clear, and thoroughly addresses the question based on the content of the document.

Output only the explanation in JSON format, with the explanation as the value of the 'explanation' key.

**Example Output (in JSON format):**
```json
{
  "explanation": "Your detailed explanation and reasoning based on the document goes here."
}
```""",
    "user": "Original document: {document}\nQuestion: {question}",
}

# GenerateAnswer do executor oficial (aspas tipográficas ’ como no original)
GENERATE_ANSWER_USER = (
    "Find the useful content from the provided documents, then answer the question. Answer the question directly. "
    "Your response should be very concise. Please provide use ’So the final answer is:’ as a prefix for the final "
    "answer. The following are given documents. \n{reference}\nAnswer the question directly. Your response should be "
    "very concise. Please provide use ’So the final answer is:’ as a prefix for the final answer. "
    "**Question**: {question}**Response**:"
)
GENERATE_ANSWER_TIPS = "Here are some tips you need to pay attention to during the generation process: {additional_instruction}"

# ============================================================================= baselines da Tabela 3
SELF_REFINE_FEEDBACK = (
    "Documents:\n{passages}\n\nQuestion: {question}\nAnswer: {answer}\n\nReview the answer against the documents. "
    "Is it correct, complete and supported by the documents? Point out concrete problems in 1-3 sentences. "
    "If the answer is correct, reply exactly: NO ISSUES"
)
SELF_REFINE_REFINE = (
    "Find the useful content from the provided documents, then answer the question.\n\n{passages}\n\n"
    "Question: {question}\nPrevious answer: {answer}\nFeedback on the previous answer: {feedback}\n\n"
    "Write an improved answer that fixes the problems. Your response should be very concise. Please use "
    "'So the final answer is:' as a prefix for the final answer."
)

METARAG_INTERNAL = (
    "Answer the question from your own knowledge, without any documents. Your response should be very concise. "
    "Please use 'So the final answer is:' as a prefix for the final answer. **Question**: {question} **Response**:"
)
METARAG_EVAL = (
    "You are monitoring a retrieval-augmented assistant.\n\nDocuments:\n{passages}\n\nQuestion: {question}\n"
    "Answer from the documents: {answer}\nAnswer from internal knowledge: {internal}\n\n"
    "Evaluate three things and reply in JSON with boolean keys:\n"
    "- docs_sufficient: do the documents contain enough information to answer the question?\n"
    "- conflict: do the documents conflict with the internal-knowledge answer?\n"
    "- reasoning_error: does the answer contain a reasoning error given the documents?"
)
METARAG_EVAL_SCHEMA = {"type": "object", "properties": {"docs_sufficient": {"type": "boolean"},
                                                        "conflict": {"type": "boolean"},
                                                        "reasoning_error": {"type": "boolean"}},
                       "required": ["docs_sufficient", "conflict", "reasoning_error"]}
METARAG_REWRITE = "Rewrite the question as a better search query to find the missing information. Reply with the query only.\nQuestion: {question}"
METARAG_REASON = (
    "Find the useful content from the provided documents, then answer the question. Think step by step, then give the "
    "answer.\n\n{passages}\n\n{hint}**Question**: {question}\nPlease use 'So the final answer is:' as a prefix for the "
    "final answer. **Response**:"
)
