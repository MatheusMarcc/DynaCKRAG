"""DynaKRAG (Wu et al., arXiv:2607.06507, 2026) — controle aprendido sobre operações atômicas de evidência.

Mapa do artigo → código:
  * Estado de evidência s_t ............................ ``state.EvidenceState``
  * 7 operações atômicas (Tabela 5) ...................... ``env.DynaEnv``
  * Camada de validade A(s_t) (Tabela 5) ................. ``validity.valid_actions``
  * Features observáveis (apêndice "Runtime features") ... ``features``
  * Alvos da Eq. (1) e treino offline .................... ``training``
  * Modelo de valor v̂θ (RF) + modelo de continuação ..... ``controller``
  * Algoritmo 1 (inferência) + compressão terminal ....... ``inference``
  * Baselines controlados (Tabela 1) ..................... ``baselines``
  * Suíte de benchmark (Tabelas 1, 2, 6, 7, 8; Figs 3–5) . ``bench`` e ``report``
"""
