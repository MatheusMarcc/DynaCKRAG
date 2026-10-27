# ragbench

Reprodução de dois métodos de *Retrieval-Augmented Generation* com modelos locais, cada um isolado, com os
baselines, as ablações e as tabelas do respectivo artigo:

- **RAG-Critic**: Dong et al., *RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval
  Augmented Generation*, ACL 2025.
- **DynaKRAG**: Wu et al., *DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop
  Retrieval-Augmented Generation*, arXiv:2607.06507, 2026.

Os dois métodos compartilham apenas a infraestrutura: cliente de LLM com cache, busca, datasets, métricas,
relatórios e um painel web local. Os modelos são servidos pelo [Ollama](https://ollama.com).

## Visão geral

| | RAG-Critic | DynaKRAG |
|---|---|---|
| Ideia | um crítico ajustado aponta o erro da resposta; um planner escreve um programa que a corrige | um controlador aprendido escolhe, a cada passo, a próxima operação de evidência |
| Implementado | Standard RAG → crítico RAG-Critic-3B (pesos oficiais) → planner → executor das 5 funções | 7 operações atômicas, camada de validade, modelo de valor (Eq. 1), modelo de continuação, compressão terminal |
| Baselines | Standard RAG, Self-Refine, FLARE, MetaRAG | Fixed-K, IRCoT, Self-Ask+Search, CRAG, Adaptive-RAG, S2G-RAG, CoRAG, PAR²-RAG |
| Ablações | sem crítico ajustado, sem auto-planejamento, 2 iterações | sem compressão, sem checagem de suficiência, sem controlador aprendido, limite de 1, 2 ou 3 buscas |
| Tabelas e figuras | Tabelas 3, 4, 5/7 e 6; Figura 6 | Tabelas 1, 2, 6, 7 e 8; Figuras 3, 4 e 5 |
| Datasets | NQ, TriviaQA, HotpotQA, 2Wiki e o RAG-Error Bench oficial | HotpotQA, 2Wiki e MuSiQue |
| Métricas | EM, F1 e Overall F1; identificação e Tag1/Tag2 F1 no RAG-Error Bench | EM, F1 e tokens por pergunta |

Cada artigo é avaliado nos próprios datasets e com as próprias métricas. A descrição de como cada parte dos
artigos foi implementada, incluindo as decisões tomadas onde o texto é omisso, está em
[`docs/RAG-Critic.md`](docs/RAG-Critic.md) e [`docs/DynaKRAG.md`](docs/DynaKRAG.md).

## Requisitos

- Python 3.10 ou superior.
- Ollama em execução na mesma máquina (ou outro servidor compatível com a API da OpenAI; ver `llm.backend`).
- GPU NVIDIA com 4 GB de VRAM para o gerador padrão (`qwen2.5:3b`). O gerador dos artigos, `qwen2.5:7b`,
  requer cerca de 8 GB.
- Windows (PowerShell) ou Linux. Acesso à internet na primeira execução, para baixar datasets e modelos.

## Instalação

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
ollama pull qwen2.5:3b
ollama pull hf.co/mradermacher/RAG-Critic-3B-GGUF:Q4_K_M
ollama pull bge-large                              # opcional: monta os índices na GPU
```

Os scripts `run_all.ps1` (Windows) e `run_all.sh` (Linux) fazem essas etapas e executam a rodada completa.

## Uso

```bash
python -m ragbench smoke                         # verificação de ponta a ponta com LLM simulado (sem GPU)
python -m ragbench doctor  --profile mini        # ambiente, modelos, uso da GPU e estimativa de tempo
python -m ragbench prepare --profile mini        # datasets, corpora, índices e RAG-Error Bench
python -m ragbench bench   --profile mini        # executa os métodos, em ordem de prioridade
python -m ragbench report  --profile mini        # gera os relatórios com o que já terminou
python -m ragbench ui                            # painel em http://127.0.0.1:8765
python -m ragbench diagnostico --profile mini    # refaz as falhas de busca e raciocínio com o gabarito no contexto
python -m ragbench llm-base --profile mini       # baseline: o mesmo gerador sem método e sem recuperação
```

O comando `smoke` deve terminar com `smoke OK — 37 jobs`. Uma execução interrompida é retomada ao repetir o
mesmo comando: as chamadas ao LLM ficam em cache e cada pergunta é gravada ao terminar.

Para usar outro modelo, por exemplo o gerador dos artigos:

```bash
python -m ragbench bench --profile standard --set models.generator=qwen2.5:7b --set models.planner=qwen2.5:7b
```

O guia de uso completo (comandos, opções, painel, execução remota e solução de problemas) está em
[`docs/USO.md`](docs/USO.md).

### Perfis

| Perfil | Perguntas por dataset | Treino do controlador (DynaKRAG) | RAG-Error Bench |
|---|---:|---:|---:|
| `smoke` | 12 (sintéticas) | 16 | 12 (sintético) |
| `mini` | 30 | 60 | 120 |
| `quick` | 100 | 150 | 380 (estratificado) |
| `standard` | 300 | 1.000 | 1.900 (completo) |

O perfil `standard` usa o tamanho de treino do controlador descrito no artigo do DynaKRAG. O comando `doctor`
estima a duração de cada perfil com a velocidade medida na máquina.

### Ordem de execução

| Tier | Conteúdo |
|---|---|
| 1 | RAG-Error Bench, Standard RAG e RAG-Critic; treino do controlador, Fixed-K e DynaKRAG |
| 2 | ablações dos dois artigos e o crítico zero-shot no benchmark |
| 3 | baselines da literatura e a varredura do limite de buscas |

### Saídas

Em `ragbench_results/<perfil>/`:

| Arquivo | Conteúdo |
|---|---|
| `TABELAS.md`, `tabelas.tex`, `figs/tabela_*.png` | uma tabela por artigo, no formato e com as métricas do artigo |
| `ragcritic/REPORT.md`, `dynakrag/REPORT.md` | tabelas e figuras de cada artigo, com os valores publicados ao lado |
| `COMPARACAO.md`, `comparacao.csv` | cada afirmação dos artigos, com a diferença medida ao lado da publicada |
| `metrics.csv` | métricas por método e dataset |
| `FALHAS.md`, `falhas_*.csv` | análise de falhas por pergunta |
| `REPORT.md` | índice e estado de cada etapa |

## Fidelidade aos artigos

**Igual aos artigos.** Prompts e funções do executor do RAG-Critic (repositório oficial), pesos oficiais do
crítico, métricas do RAG-Error Bench portadas do script oficial; no DynaKRAG, as sete operações, o alvo de
treino da Eq. 1, a regra de escolha da Eq. 2, os hiperparâmetros do controlador, o retriever e os parâmetros
de compressão.

**Adaptado por configuração.** Estes itens diferem dos artigos nos perfis padrão e podem ser alterados:

| Item | Artigos | Padrão do pacote |
|---|---|---|
| Gerador | Qwen2.5-7B e outros | `qwen2.5:3b` (`models.generator`) |
| Planner do RAG-Critic | Llama3.1-70B ou Qwen2.5-72B | `qwen2.5:3b` (`models.planner`) |
| Perguntas de avaliação | 100 por dataset (RAG-Critic); splits completos (DynaKRAG) | 30, 100 ou 300, conforme o perfil |
| Treino do controlador | 1.000 perguntas | 60, 150 ou 1.000, conforme o perfil |

**Diferente dos artigos no código atual.**

- *Corpus.* Os artigos buscam na Wikipedia completa. O pacote monta, para cada dataset, um corpus com os
  parágrafos que acompanham as perguntas amostradas. A busca fica mais fácil do que nos artigos, o que afeta
  principalmente os baselines de busca estática.
- *DynaKRAG.* O artigo não publica código nem prompts. A regra de parada, o prompt da checagem de suficiência,
  o detector de entidade-ponte e o K do Fixed-K foram reconstruídos a partir do texto e das tabelas; cada
  decisão está registrada em `docs/DynaKRAG.md`.
- *RAG-Critic.* Três datasets de resposta longa (ASQA, ELI5, WoW), o baseline Self-RAG e duas ablações que
  exigem retreinar o crítico não estão implementados.

**Não consta nos artigos.** O pacote inclui cinco recursos de apoio, identificados como tais nos relatórios: a
análise de falhas por pergunta (`FALHAS.md`), o diagnóstico das falhas de busca e de raciocínio (`DIAGNOSTICO.md`),
o baseline LLM base (`LLM_BASE.md`), a auditoria de formato do RAG-Error Bench e o painel.

## Estrutura

```
ragbench/
  core/        cliente de LLM com cache, execução retomável, datasets, busca, modelo simulado
  ragcritic/   RAG-Critic: fluxo, crítico, planner, executor, baselines, RAG-Error Bench, treino do crítico
  dynakrag/    DynaKRAG: estado, operações, validade, controlador, treino, baselines
  configs/     configuração base e perfis
  ui/          painel web local
  analysis.py  análise de falhas        compare.py  comparação com os artigos
  tables.py    tabelas de apresentação  report.py   índice dos relatórios
docs/          notas de implementação por artigo e guia de uso
tests/         testes automatizados
```

## Testes

```bash
python -m pytest -q tests        # 107 testes
python -m ragbench smoke         # 37 etapas com LLM simulado
```

Dois testes são ignorados quando faltam os arquivos oficiais do RAG-Error Bench (variável
`RAGBENCH_ERRORBENCH_DIR`) ou o Node.js.

## Créditos e licenças de terceiros

O pacote reproduz prompts e métricas do repositório oficial do RAG-Critic e usa modelos e datasets de
terceiros, baixados em tempo de execução. Ver [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
