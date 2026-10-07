"""Construção do sistema hierárquico de erros (Seção 3.1 do RAG-Critic), em escala reduzida.

Etapa 1 — amostragem de respostas com erro: um POOL de LLMs responde perguntas de vários datasets com o
          Prompt A; um modelo supervisor julga cada resposta contra o gold com o Prompt B (análise do erro).
Etapa 2 — anotação open-set: o supervisor dá tags livres para cada análise (Prompt C); normalização
          (remove rótulos com frequência < α e com mais de 25 tokens; descarta saídas fora do JSON).
Etapa 3 — sumarização: embeddings BGE-M3 → padronização → clustering hierárquico de Ward em 20 grupos;
          um LLM nomeia cada grupo (nível 2); a atribuição ao nível 1 é sugerida automaticamente e deve ser
          revisada por humanos (no artigo: três anotadores). Por fim, mapeamento reverso → D_Error^QA (Eq. 1).

Artigo × aqui: 15 LLMs de 3B–70B e Qwen2.5-72B como supervisor → pool de modelos pequenos do Ollama e um
3B/7B como supervisor. O que se reproduz é o PROCEDIMENTO; a taxonomia oficial (7/19/4000+) continua
sendo a referência para o benchmark.
"""
