# dynacritic — DynaKRAG + RAG-Critic num notebook com RTX 3050 (4 GB)

Framework que junta os dois artigos num único pipeline:

```
pergunta ─► [Laço A · DynaKRAG]  camada de validade → política (uniforme | random forest) → ação → novo estado … parar
         ─► compressão terminal ─► gerador ─► resposta
         ─► [RAG-Critic-3B]  Correct → fim
                             Error   → tags + análise ─► [Laço B · reparo: menu | code | state] ─► resposta corrigida
```

Tudo roda com **Ollama** na sua máquina. Uma API externa só é necessária se você quiser
trocar para modelos maiores depois: basta mudar `llm.backend` para `openai`.

---

## 1. O que roda no seu hardware

| Peça | Como roda aqui | Observação |
|---|---|---|
| Gerador | `qwen2.5:3b` no Ollama (~2 GB, cabe inteiro na GPU) | o artigo usa 7B; o `qwen2.5:7b` roda com parte na CPU (bem mais lento) |
| Crítico | `hf.co/mradermacher/RAG-Critic-3B-GGUF:Q4_K_M` | é o crítico **oficial**, só convertido para GGUF — não precisa treinar nada |
| Planner (reparo) | `qwen2.5:3b` com saída JSON (modo `menu`) | modelos de 3B erram muito escrevendo código livre |
| Controlador | `RandomForestRegressor` (sklearn), CPU | não é LLM; treina em segundos |
| Retrieval | BM25 na CPU (padrão) ou denso (`bge-small`/`bge-large`) na CPU | a GPU fica só para o LLM |
| Corpus | união dos parágrafos das perguntas amostradas do HotpotQA | indexar a Wikipedia inteira não cabe num notebook |

**O que muda em relação aos artigos:** corpus reduzido, gerador 3B e amostras menores
(300 + 300 por padrão). Com isso os números absolutos **não** batem com o artigo.
As comparações relativas continuam válidas: uniforme × RF, com/sem compressão, com/sem crítico
e os três modos de reparo, porque todas rodam no mesmo corpus, com o mesmo modelo e o mesmo código.

---

## 2. Instalação (Windows)

1. Instale o **Ollama** para Windows (ollama.com) e deixe o app aberto.
2. Baixe os modelos (PowerShell):
   ```powershell
   ollama pull qwen2.5:3b
   ollama pull hf.co/mradermacher/RAG-Critic-3B-GGUF:Q4_K_M
   ```
3. Python 3.10+ num ambiente virtual, dentro desta pasta:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   (No Linux/WSL: `source .venv/bin/activate`.)

---

## 3. Passo a passo

Rode os comandos **nesta ordem**, a partir da raiz do projeto. Cada etapa grava em `data/`
e tem um critério para você saber se deu certo.

| # | Comando | O que conferir |
|---|---|---|
| 0 | `python -m dynacritic smoke` | termina com `smoke OK` (sem GPU, sem Ollama: dados sintéticos + LLM falso) |
| 1 | `python -m dynacritic doctor` | os 3 modelos com `[ok]`, tok/s e **% na GPU** (o 3B deve ficar ~100%) |
| 2 | `python -m dynacritic prepare` | `train=300 dev=300 corpus≈6000 parágrafos` |
| 3 | `python -m dynacritic index` | índice BM25 salvo |
| 4 | `python -m dynacritic baseline --k 5 --limit 30` | EM/F1 plausíveis; depois rode sem `--limit` |
| 5 | `python -m dynacritic run --policy uniform` | a ablação "sem controlador aprendido" |
| 6 | `python -m dynacritic label` | nº de pares (estado, ação) rotulados |
| 7 | `python -m dynacritic train` | MAE por ação e as features mais importantes |
| 8 | `python -m dynacritic run --policy rf` e `run --policy rf --no-compress` | RF > uniforme; compressão ajuda |
| 9 | `python -m dynacritic critic-bench --file <RAG-Critic>/rag_error_bench/test_data/critic_model_test.json --n 200` | identificação alta nos dois lados (o artigo reporta ~96%) |
| 10 | `python -m dynacritic critic --run dyna_rf` | quantas respostas o crítico marcou e a concordância com o EM |
| 11 | `python -m dynacritic repair --run dyna_rf --mode menu` (e `code`, `state`) | F1 depois do reparo × F1 antes |
| 12 | `python -m dynacritic eval` | tabela com EM, F1, SR, tokens, chamadas e passos + composição das ações |

Dicas:
- Use `--limit 20` em qualquer comando para um teste rápido antes da rodada completa.
- Qualquer chave do `config.yaml` pode ser sobrescrita: `--set models.generator=qwen2.5:7b`.
- Todas as chamadas ao LLM ficam em cache (`data/llm_cache.sqlite`). Repetir um experimento
  com a mesma configuração não gasta nada.
- Para o passo 9, clone o repositório oficial: `git clone https://github.com/RUC-NLPIR/RAG-Critic`.

### Quanto tempo leva
O `doctor` mostra os tok/s da sua máquina. Como referência, o laço A faz cerca de 3–4 chamadas ao
LLM por pergunta, a rotulagem cerca de 2–4, o crítico 1 e o reparo 2–4 (só nas respostas marcadas
como erro). Com um 3B inteiro na GPU, conte algo na faixa de 10–30 s por pergunta no laço A.
Com 300 perguntas, isso dá de 1 a 3 horas por rodada, e o cache torna as repetições instantâneas.
**Meça antes de rodar tudo:** faça `--limit 20` e multiplique.

### Por que as etapas são separadas
Com 4 GB de VRAM, o gerador e o crítico não ficam os dois carregados na GPU. Por isso o pipeline
roda **por estágio**: primeiro o laço A em todas as perguntas, depois o crítico em todas, depois o reparo.
Assim o Ollama troca de modelo 3 vezes, em vez de 3 vezes por pergunta.

---

## 4. Como cada artigo foi implementado

### DynaKRAG (laço A) — `env.py`, `validity.py`, `features.py`, `controller.py`, `labeling.py`
- **Estado** (`state.py`): pergunta, ranking inicial/fronteira, docs, query ativa, lacuna, resultado
  de suficiência, histórico, orçamentos.
- **6 ações + compressão**: `retrieve_more`, `gap_query`, `rewrite_query`, `bridge_entity_expand`,
  `sufficiency_check`, `stop_answer` e `compress_answer_evidence` no final.
- **Camada de validade** = Tabela 5 do artigo, aplicada antes de qualquer pontuação.
- **Alvo** = Equação 1: ΔSR para aquisição, `1[SR=1]` para suficiência e `SR` para parar.
  O gold só aparece aqui.
- **RF**: 300 árvores, folha mínima 8, semente 13, λ = 0 (igual ao artigo).
- Limites: 4 passos, 3 docs por ação, 12 docs no contexto final (igual ao artigo).

### RAG-Critic (laço B) — `critic.py`, `repair.py`
- Prompt do crítico **idêntico byte a byte** ao formato de treino: conferido em 300 exemplos do
  benchmark oficial. O prompt vai na mensagem de sistema, com a de usuário vazia, como no repo.
- Tabela erro→ação (Tabela 11) usada como dica para o planner.
- **Modo `code`** (fiel ao artigo): o planner escreve Python com `Retrieval`, `RewriteQuery`,
  `DecomposeQuery`, `RefineDoc` e `GenerateAnswer`. O repositório oficial roda o código com `exec()`
  sem restrição. Aqui ele passa antes por uma **whitelist de AST**: sem imports, sem atributos,
  sem laços e só as 5 funções.
- **Modo `menu`**: o planner escolhe até 3 passos de um menu fixo via *structured outputs* (JSON schema).
- **Modo `state`**: a crítica vira a lacuna do estado do DynaKRAG, e o **mesmo** controlador decide o
  reparo, com passos extras. Nenhum dos dois artigos faz isso.

---

## 5. Decisões onde os artigos são vagos (documente isso no seu relatório)

1. **Política que gera os estados de treino**: uniforme sobre ações válidas, sem parar antes do limite.
2. **Detector de entidade-ponte**: nomes próprios citados nos docs que ainda não têm documento próprio.
3. **Portão de continuação**: regra `sufficient_only` (só para se a última checagem disser
   "suficiente") no lugar do modelo de continuação aprendido. Sem esse portão, o argmax para cedo
   demais, porque parar vale `SR(s)` e adquirir vale `ΔSR`.
4. **SR por título**: fração dos títulos de suporte presentes no estado, com títulos normalizados.
5. **Corpus reduzido** e BM25 como retriever padrão (o artigo usa BGE-large + FAISS no fullwiki).

---

## 6. Próximos passos sugeridos

- Trocar o portão por um **modelo de continuação aprendido**. Rótulo: se continuar até o limite
  melhora o F1 em relação a parar agora.
- `retriever.kind: dense` com `bge-large-en-v1.5` (requer `sentence-transformers` e `faiss-cpu`).
- Adicionar 2Wiki e MuSiQue (MuSiQue: `compress_max_snippets: 12`, `compress_max_tokens: 384`).
- Rodar a configuração final com um 7B, via Ollama com offload, numa GPU alugada ou numa API
  (`llm.backend: openai`). O cache e o código continuam os mesmos.
- Testar λ > 0 no controlador, com custo explícito por ação.

---

## 7. Estrutura

```
dynacritic/
  cli.py         comandos
  config.py      config.yaml + --set
  data.py        HotpotQA → registros + corpus reduzido
  retrieval.py   BM25 / denso
  llm.py         Ollama / OpenAI-compatível / mock, com cache e contagem de tokens
  prompts.py     todos os prompts
  state.py       estado de evidências
  env.py         ações atômicas do DynaKRAG + detector de ponte
  validity.py    camada de validade A(s)
  features.py    features observáveis do par (estado, ação)
  controller.py  política uniforme, RF e portão de continuação
  loop.py        laço A, compressão, resposta e baseline de K fixo
  labeling.py    rotulagem ΔSR + treino da RF
  critic.py      RAG-Critic-3B + validação no RAG-Error Bench
  repair.py      laço B: menu / code (sandbox) / state
  evaluate.py    tabela comparativa
  mock.py, toy.py  LLM falso + dados sintéticos para o smoke test
tests/test_core.py  python -m pytest -q
```
