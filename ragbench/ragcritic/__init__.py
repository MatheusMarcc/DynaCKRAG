"""RAG-Critic (Dong et al., ACL 2025) — fluxo agêntico guiado por um crítico de erros.

Mapa do artigo → código:
  * Sistema hierárquico de erros, Tabelas 11 e 12 ...... ``taxonomy``
  * Prompts A–F + prompts oficiais das 5 funções ....... ``prompts``
  * Gerador π_α (Standard RAG, Prompt A) ................ ``workflow.stage_generate``
  * Crítico π_θ (RAG-Critic-3B / LLM zero-shot) ........ ``critic``
  * Planner π_β (Prompt F → código Python) ............. ``planner``
  * Executor (5 funções, sandbox) ....................... ``executor``
  * Algoritmo 1 + iterações N ........................... ``workflow``
  * Baselines (Self-Refine, FLARE, MetaRAG) ............. ``baselines``
  * RAG-Error Bench (Tabelas 5/7, Fig. 5/7) + auditoria . ``errorbench``
  * Construção do sistema de erros (etapas 1–3) ......... ``errorsystem``
  * Alinhamento do crítico (SFT + DPO coarse-to-fine) ... ``training``
"""
