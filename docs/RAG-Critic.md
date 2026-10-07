# RAG-Critic — notas de implementação

Dong et al., *RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval Augmented
Generation*, ACL 2025. Código oficial: <https://github.com/dongguanting/RAG-Critic> (licença MIT). Os prompts
e as funções do executor seguem esse repositório.

## Mapa artigo → código (`ragbench/ragcritic/`)

| Artigo | Código | Observação |
|---|---|---|
| Sistema de erros 7 → 19 → 4000+ | `taxonomy.TIER1/TIER2/TIER2_PARENT` | de `all_tags_structure_final.json` |
| Tabela 11 (erro → ações) | `taxonomy.ERROR_ACTION_TABLE` | também gera os programas da ablação por regra |
| Tabela 12 (5 funções, 15 sub-ações) | `taxonomy.ACTION_FUNCTIONS` e `executor.py` | |
| Prompt A (geração) | `prompts.PROMPT_A` | Standard RAG com top-5 |
| Prompts B e C (análise de erro e tags open-set) | `prompts.PROMPT_B/PROMPT_C` | etapas 1–2 do sistema de erros |
| Prompt D (crítico ajustado) e E (baselines no benchmark) | `prompts.render_critic` | reproduz os 3.800 prompts do benchmark **byte a byte** |
| Prompt F (planner) | `prompts.PLANNER_SYSTEM/USER` | texto de `critic_agent/prompt_template.py` |
| π_θ: RAG-Critic-3B | `critic.Critic(mode="finetuned")` | pesos oficiais em GGUF, ChatML cru |
| π_β: planner (Eq. 2) | `planner.plan` | recebe y completo e a tabela T |
| Executor | `executor.Executor` | funções oficiais + sandbox |
| Algoritmo 1 + N iterações (Tabela 6) | `workflow.run_variant` | executado por estágio |
| Baselines da Tabela 3 | `baselines.py` | Self-Refine, FLARE, MetaRAG |
| RAG-Error Bench (Tabelas 5/7) | `errorbench.py` | métricas portadas de `caculate_acc.py` e verificadas |
| Etapas 1–3 (Seção 3.1) | `errorsystem/pipeline.py` | opcional, em escala reduzida |
| SFT + DPO coarse-to-fine (Seção 3.2) | `training/` | scripts TRL; treinar o modelo de 3B requer GPU de maior porte |

## Configuração (artigo × pacote)

| | Artigo | ragbench |
|---|---|---|
| Gerador π_α | Llama3.1-8B/70B, Qwen2.5-7B/72B | `qwen2.5:3b` |
| Crítico π_θ | RAG-Critic-3B (Qwen2.5-3B, SFT + DPO) | os **mesmos pesos** (GGUF Q4_K_M de mradermacher) |
| Planner π_β | Llama3.1-70B / Qwen2.5-72B | `qwen2.5:3b` (configurável em `models.planner`) |
| Retriever | E5-base-v2 no wiki-2018 (21M passagens, FlashRAG) | E5-base-v2 no corpus reduzido de cada dataset |
| Documentos | top-5 | top-5 |
| Amostra | 100 perguntas por dataset | 100 (`quick`) |
| RAG-Error Bench | 1.900 exemplos | 380 estratificados (`quick`) ou 1.900 (`standard`) |

## Detalhes que fazem diferença

* **Formato do crítico.** O `demo.ipynb` oficial chama o modelo com o system padrão do Qwen ("You are Qwen,
  created by Alibaba Cloud...") e o prompt D como mensagem de usuário. O pacote monta o ChatML explicitamente e
  chama o `/api/generate` do Ollama com `raw=true`, para não depender do template que o Ollama infere do GGUF.
* **Artefato de formato do benchmark.** No `critic_model_test.json` e no `baseline_test.json` oficiais, os
  950 exemplos "Correct" usam recuo de 20 espaços e os 950 "Error", 16. Além da avaliação descrita no artigo,
  o `errorbench` avalia o crítico com o recuo trocado (a função `swap_indent` re-renderiza o mesmo exemplo
  sem mudar nenhuma palavra). Essa auditoria não consta no artigo. As linhas "todos com 20/16 espaços" do
  relatório combinam as duas rodadas e simulam um benchmark sem o artefato. O `build-train-data` neutraliza o
  recuo por padrão, para que um crítico treinado com esses dados não aprenda o atalho.
* **Recuo no fluxo agêntico.** O recuo do prompt do crítico no fluxo completo é definido por
  `ragcritic.critic_template` (padrão: 20 espaços). O `plan_agent.py` oficial lê saídas do crítico já
  prontas, de modo que o recuo usado nos experimentos do artigo não é conhecido.
* **Métricas do benchmark.** `official_metrics` é um porte fiel de `rag_error_bench/caculate_acc.py`,
  inclusive nas peculiaridades: pula os "Correct" acertados, conta as tags de exemplos com julgamento
  errado e calcula F1 micro. A verificação com 1.900 predições aleatórias dá resultado **idêntico** ao do
  script oficial em todas as métricas globais e por categoria. Na Tabela 5 do artigo, o "Avg" de
  classificação (58,3) não é a média de Tag1 (65,2) e Tag2 (42,4). Por isso o relatório traz F1 e acurácia
  de cada nível separadamente.
* **Diferenças deliberadas em relação ao código oficial** (todas registradas):
  * `GenerateAnswer`: o oficial não formata a dica (manda a string literal `{additional_instruction}`).
    No pacote a dica é formatada.
  * `RewriteQuery` com instrução livre: o prompt oficial tem chaves JSON sem escape e o `.format` quebra.
    No pacote as chaves estão escapadas.
  * Falha na execução do programa: o oficial usa resposta vazia (conta como erro). O pacote mantém a
    resposta anterior (`on_exec_fail: keep_previous`; `empty` reproduz o comportamento oficial) e o
    relatório informa a taxa de falhas.
  * O código do planner roda numa sandbox (whitelist de AST, orçamento de 12 chamadas, laços limitados),
    e não em `exec()` livre.
  * Extração da resposta curta: a mesma função tolerante para todos os métodos. O oficial faz
    `split("So the final answer is:")[-1]`.
* **Planner.** A Eq. (2) do artigo dá ao planner (q, D_q, p, y, F, T), ou seja, a crítica completa e a
  tabela T. O `plan_agent.py` oficial passa só as tags de nível 2. O pacote segue o artigo.
* **Execução por estágio.** Para que o gerador e o crítico não precisem ficar carregados ao mesmo tempo,
  cada dataset roda G para todas as perguntas, depois C para todas, depois P e E. São poucas trocas de
  modelo em vez de duas por pergunta.

## Ablações (Tabelas 4 e 6)

| Linha | Variante | Como |
|---|---|---|
| w/o Critic Model | `ragcritic_zeroshot_critic` | o LLM base faz o papel de crítico com o Prompt E |
| w/o Auto-Planning | `ragcritic_rule_planning` | programa fixo por tipo de erro de nível 1, pela tabela T |
| w/o Data-driven / w/o Manual Sum. | — | exigem retreinar o crítico com outra taxonomia (scripts em `training/`) |
| Tabela 6 | `ragcritic_n2` | duas iterações do fluxo |

## Datasets

* **HotpotQA e 2Wiki**: os mesmos corpora reduzidos usados no DynaKRAG.
* **NQ e TriviaQA**: passagens do Wikipedia-DPR (100 palavras) da versão Tevatron, positivas + até 20
  negativas por pergunta, reunidas num corpus. Alternativas automáticas: `florin-hf/nq_open_gold` e as
  páginas do TriviaQA rc.wikipedia cortadas em 100 palavras. Os dados pré-processados do repositório
  oficial (top-5 do E5 no wiki-2018) podem ser apontados em `data.local` com `format: ragcritic`.
* **ASQA, ELI5 e WoW** (respostas longas) não estão implementados.

## Treinar um crítico (opcional)

```powershell
pip install -r requirements-train.txt      # + torch com CUDA
python -m ragbench ragcritic build-train-data --source official --limit 20000
python -m ragbench ragcritic train sft -- --data ragbench_data\critic_training\sft_official.jsonl `
    --model Qwen/Qwen2.5-0.5B-Instruct --out ragbench_data\critic_training\sft-0.5b --qlora --max-samples 4000
python -m ragbench ragcritic train dpo -- --data ragbench_data\critic_training\dpo_official.jsonl `
    --model ragbench_data\critic_training\sft-0.5b --out ragbench_data\critic_training\dpo-0.5b --qlora
python -m ragbench.ragcritic.training.export --adapter ragbench_data\critic_training\dpo-0.5b --base ... --out ...
```

Os hiperparâmetros padrão são os do apêndice B.2 (SFT: lr 7e-6, 3 épocas, batch 128; DPO: lr 5e-7,
β 0,3, sigmoid, 2 épocas, batch 64). Em GPUs de 4 GB só cabe um modelo pequeno com QLoRA; o modelo de 3B do
artigo requer GPU de maior porte.
