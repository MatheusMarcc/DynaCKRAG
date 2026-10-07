"""Números publicados nos dois artigos, para comparação lado a lado nos relatórios.

Conferidos contra os PDFs em 05/10/2026 (DynaKRAG: Tabelas 1 (Qwen2.5-7B), 2, 6, 7, 8 e Fig. 5; RAG-Critic:
Tabelas 3, 4, 5, 6 e 7).

Atenção ao comparar: os artigos usam modelos 7B–72B, corpora completos (fullwiki / wiki-2018) e splits
inteiros. Com outro modelo, corpus reduzido ou amostra, compare TENDÊNCIAS (quem ganha
de quem, o efeito de cada componente), não os valores absolutos.
"""

# ----------------------------------------------------------------------------- DynaKRAG — Tabela 1 (Qwen2.5-7B)
DYNAKRAG_TABLE1_QWEN7B = {  # método: {dataset: (EM, F1, Tok)}
    "Fixed-K": {"hotpotqa": (0.3958, 0.5177, 1892.3), "2wiki": (0.3177, 0.3757, 3088.9), "musique": (0.1518, 0.2553, 2918.3)},
    "IRCoT": {"hotpotqa": (0.4339, 0.5617, 1723.5), "2wiki": (0.3514, 0.4308, 2441.8), "musique": (0.1862, 0.2889, 2109.6)},
    "S2G-RAG": {"hotpotqa": (0.4425, 0.5710, 2807.3), "2wiki": (0.3818, 0.4621, 3543.5), "musique": (0.1936, 0.2999, 3886.8)},
    "CoRAG-s3": {"hotpotqa": (0.4282, 0.5573, 2814.3), "2wiki": (0.3490, 0.4273, 3675.8), "musique": (0.1858, 0.2848, 3263.6)},
    "Adaptive-RAG": {"hotpotqa": (0.3862, 0.5042, 1596.4), "2wiki": (0.2976, 0.3557, 2180.0), "musique": (0.1647, 0.2579, 2160.4)},
    "PAR²-RAG": {"hotpotqa": (0.3779, 0.4952, 3053.6), "2wiki": (0.3071, 0.3691, 3971.2), "musique": (0.1448, 0.2480, 3755.6)},
    "CRAG": {"hotpotqa": (0.4181, 0.5445, 1720.1), "2wiki": (0.3437, 0.4099, 2375.4), "musique": (0.1676, 0.2740, 2479.5)},
    "Self-Ask+Search": {"hotpotqa": (0.4043, 0.5273, 3302.0), "2wiki": (0.3494, 0.4345, 4565.5), "musique": (0.1974, 0.2998, 4761.8)},
    "DynaKRAG": {"hotpotqa": (0.4732, 0.5998, 2373.1), "2wiki": (0.4587, 0.5340, 3077.9), "musique": (0.1986, 0.3061, 3082.3)},
}

# Tabela 2 — F1 dos componentes (Qwen2.5-7B)
DYNAKRAG_TABLE2 = {
    "DynaKRAG": {"hotpotqa": 0.5998, "2wiki": 0.5340, "musique": 0.3061},
    "w/o terminal compression": {"hotpotqa": 0.5826, "2wiki": 0.4963, "musique": 0.2978},
    "w/o sufficiency check": {"hotpotqa": 0.5835, "2wiki": 0.4623, "musique": 0.2800},
    "w/o learned controller": {"hotpotqa": 0.5519, "2wiki": 0.4762, "musique": 0.2665},
    "Fixed-K static reference": {"hotpotqa": 0.5177, "2wiki": 0.3757, "musique": 0.2553},
}

# Tabela 6 — varredura do limite de chamadas de retrieval: (EM, F1, Tokens, Ret. calls, LLM calls, Exhausted)
DYNAKRAG_CAP_SWEEP = {
    "hotpotqa": {1: (0.3801, 0.4938, 1211.24, 1.000, 3.000, 1.0000), 2: (0.4590, 0.5823, 1648.15, 1.738, 3.100, 0.7376),
                 3: (0.4731, 0.5996, 2373.78, 2.437, 3.663, 0.6995)},
    "2wiki": {1: (0.3628, 0.4111, 1325.99, 1.000, 3.000, 1.0000), 2: (0.4412, 0.5096, 1948.57, 1.804, 3.006, 0.8038),
              3: (0.4587, 0.5340, 3077.82, 2.598, 3.687, 0.7945)},
    "musique": {1: (0.1080, 0.1970, 1502.30, 1.000, 3.000, 1.0000), 2: (0.1791, 0.2796, 2084.88, 1.947, 3.007, 0.9470),
                3: (0.1643, 0.2641, 3137.42, 2.889, 3.690, 0.9417)},
}

# Tabela 7 — contagem média de ações por exemplo
DYNAKRAG_ACTIONS = {
    "hotpotqa": {"retrieve_more": 1.072, "sufficiency_check": 0.998, "gap_query": 1.241, "rewrite_query": 0.106,
                 "bridge_entity_expand": 0.021, "compress": 1.000, "ret_calls": 2.439, "llm_calls": 3.661},
    "2wiki": {"retrieve_more": 1.132, "sufficiency_check": 0.999, "gap_query": 1.440, "rewrite_query": 0.009,
              "bridge_entity_expand": 0.018, "compress": 1.000, "ret_calls": 2.599, "llm_calls": 3.686},
    "musique": {"retrieve_more": 1.249, "sufficiency_check": 0.977, "gap_query": 1.543, "rewrite_query": 0.009,
                "bridge_entity_expand": 0.112, "compress": 1.000, "ret_calls": 2.912, "llm_calls": 3.664},
}

# Tabela 8 — diagnóstico da sonda de suficiência: (passos completo, passos sem sonda, F1 completo, F1 sem sonda)
DYNAKRAG_PROBE = {"hotpotqa": (3.437, 4.000, 0.5998, 0.5835), "2wiki": (3.598, 4.000, 0.5340, 0.4623),
                  "musique": (3.890, 4.000, 0.3061, 0.2800)}

# Fig. 5 — ΔF1 (DynaKRAG completo − comparador) por estrutura de pergunta no 2Wiki
DYNAKRAG_FIG5 = {
    "compositional": {"Fixed-K": 24.3, "Uniform": 14.0, "No suff.": 16.9, "No comp.": 1.2},
    "bridge_comparison": {"Fixed-K": 13.0, "Uniform": -2.3, "No suff.": -0.6, "No comp.": 4.9},
    "comparison": {"Fixed-K": 7.9, "Uniform": -0.3, "No suff.": -0.9, "No comp.": 7.8},
    "inference": {"Fixed-K": 7.7, "Uniform": 4.1, "No suff.": 4.1, "No comp.": 2.5},
}

# ----------------------------------------------------------------------------- RAG-Critic — Tabela 3 (F1; EM onde há)
RAGCRITIC_TABLE3 = {  # (backbone, método): {dataset: (EM, F1)}
    ("Qwen2.5-7B", "Standard RAG"): {"nq": (18, 33.2), "triviaqa": (41, 52.1), "hotpotqa": (32, 43.5), "2wiki": (21, 28.2), "overall_f1": 30.3},
    ("Qwen2.5-7B", "RAG-Critic"): {"nq": (22, 37.3), "triviaqa": (53, 58.5), "hotpotqa": (37, 48.8), "2wiki": (25, 32.8), "overall_f1": 34.1},
    ("Llama3.1-8B", "Standard RAG"): {"nq": (23, 38.3), "triviaqa": (44, 55.3), "hotpotqa": (27, 35.2), "2wiki": (18, 30.1), "overall_f1": 28.7},
    ("Llama3.1-8B", "Self-Refine"): {"nq": (10, 22.3), "triviaqa": (20, 32.3), "hotpotqa": (15, 24.7), "2wiki": (12, 23.1), "overall_f1": 22.2},
    ("Llama3.1-8B", "FLARE"): {"nq": (12, 19.8), "triviaqa": (48, 57.1), "hotpotqa": (20, 25.8), "2wiki": (10, 22.7), "overall_f1": 22.8},
    ("Llama3.1-8B", "Self-RAG"): {"nq": (27, 32.3), "triviaqa": (24, 35.7), "hotpotqa": (9, 17.6), "2wiki": (4, 18.9), "overall_f1": 24.7},
    ("Llama3.1-8B", "MetaRAG"): {"nq": (22, 40.2), "triviaqa": (50, 59.2), "hotpotqa": (37, 47.0), "2wiki": (20, 29.2), "overall_f1": 30.6},
    ("Llama3.1-8B", "RAG-Critic"): {"nq": (27, 42.0), "triviaqa": (50, 60.1), "hotpotqa": (40, 51.2), "2wiki": (21, 33.1), "overall_f1": 34.0},
}
# Tabela 4 — ablação (Llama3.1-8B), F1
RAGCRITIC_TABLE4 = {
    "RAG-Critic": {"nq": 42.0, "triviaqa": 60.1, "hotpotqa": 51.2},
    "w/o Data-driven": {"nq": 38.7, "triviaqa": 58.5, "hotpotqa": 48.8},
    "w/o Manual Sum.": {"nq": 40.1, "triviaqa": 59.2, "hotpotqa": 47.0},
    "w/o Auto-Planning": {"nq": 39.2, "triviaqa": 57.2, "hotpotqa": 45.5},
    "w/o Critic Model": {"nq": 37.0, "triviaqa": 56.5, "hotpotqa": 47.0},
}
# Tabela 5/7 — RAG-Error Bench: Identificação (Correct, Error, Avg) e Classificação (Tag1, Tag2, Avg)
RAGCRITIC_ERRORBENCH = {
    "o1-preview": (79.0, 59.4, 69.2, 23.6, 7.4, 15.5),
    "GPT-4o": (77.9, 78.1, 78.0, 38.5, 15.4, 26.9),
    "Claude 3.5": (46.7, 89.3, 68.2, 32.2, 10.6, 21.3),
    "Qwen2.5-72B": (79.8, 79.8, 79.8, 45.5, 17.4, 31.5),
    "Llama3.1-70B": (95.2, 42.7, 68.9, 25.7, 10.0, 17.8),
    "Qwen2.5-7B": (58.5, 85.6, 72.3, 41.8, 21.9, 31.9),
    "Llama3.1-8B": (98.7, 27.8, 63.3, 26.0, 18.4, 22.2),
    "Phi-3.5-mini": (46.5, 8.37, 27.4, 27.9, 12.6, 20.3),
    "RAG-Critic (3B)": (95.8, 96.6, 96.2, 65.2, 42.4, 58.3),
}
# Tabela 6 — iterações (F1)
RAGCRITIC_ITERATIONS = {
    "Llama3-8B": {0: {"nq": 38.3, "triviaqa": 55.3}, 1: {"nq": 42.0, "triviaqa": 60.1}, 2: {"nq": 42.4, "triviaqa": 60.4}},
}
# Fig. 6 — distribuição dos erros de nível 1 (Qwen2.5-7B + Llama3.1-8B, 9 datasets)
RAGCRITIC_FIG6 = {"Inaccurate Response": 23.51, "Incomplete Response": 21.33, "Incomplete Information": 21.18,
                  "Irrelevant Information": 17.14, "Off-Topic Response": 13.85, "Erroneous Information": 2.93,
                  "Overly Verbose Response": 0.05}
