# Guia de uso

## 1. Execução completa

Os scripts de entrada criam o ambiente virtual, instalam as dependências, conferem os modelos no Ollama e
executam `doctor → prepare → bench → report`.

Windows (PowerShell, na pasta do projeto):

```powershell
.\run_all.ps1 -Perfil smoke        # verificação sem Ollama (LLM simulado)
.\run_all.ps1 -Perfil mini -Tier 1 # 30 perguntas por dataset, só o tier 1
.\run_all.ps1                      # perfil quick, os dois artigos
.\run_all.ps1 -Paper dynakrag      # um artigo: ragcritic | dynakrag
.\run_all.ps1 -Painel              # abre o painel (equivale a painel.bat)
```

Linux:

```bash
./run_all.sh mini            # perfil mini, os dois artigos
./run_all.sh quick dynakrag  # um artigo
./run_all.sh painel          # abre o painel
```

No Windows, o script também impede a suspensão automática durante a rodada. Se o PowerShell bloquear o
script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Os scripts usam os modelos padrão do pacote. Para outro modelo, usar os comandos da seção 2 com `--set`.

## 2. Comandos

```bash
python -m ragbench smoke                                   # ponta a ponta com LLM simulado
python -m ragbench doctor  --profile quick                 # ambiente, modelos, uso da GPU, estimativa
python -m ragbench prepare --profile quick                 # datasets, corpora, índices, RAG-Error Bench
python -m ragbench bench   --profile quick                 # todos os métodos, por tier
python -m ragbench bench   --profile quick --paper dynakrag --tier 1
python -m ragbench bench   --profile quick --datasets hotpotqa --methods dynakrag/dynakrag dynakrag/fixed_k6
python -m ragbench bench   --profile quick --dry-run       # lista as etapas sem executar
python -m ragbench report  --profile quick                 # relatórios com o que já terminou
python -m ragbench analyze --profile quick                 # só a análise de falhas
python -m ragbench ui                                      # painel
python -m ragbench ui --profile mini --export painel.html  # painel em um arquivo único

# um método isolado
python -m ragbench dynakrag train --profile mini --datasets hotpotqa
python -m ragbench dynakrag run   --profile mini --dataset hotpotqa --method dynakrag --limit 10
python -m ragbench ragcritic run  --profile mini --dataset nq --method ragcritic --limit 10
python -m ragbench ragcritic errorbench --profile quick
python -m ragbench ragcritic errorbench --profile quick --which zeroshot --model llama3.2:3b

# partes opcionais do RAG-Critic
python -m ragbench ragcritic errorsystem --profile quick         # etapas 1 a 3: sistema de erros local
python -m ragbench ragcritic build-train-data --source official  # dados de SFT e DPO
python -m ragbench ragcritic train sft -- --data ... --model ...  # ver docs/RAG-Critic.md
```

O comando `bench` recusa nomes de método ou de dataset inexistentes e lista os nomes válidos. Se o Ollama
local não estiver em execução na primeira chamada ao modelo, o pacote tenta iniciá-lo (`llm.autostart`); se
não conseguir, encerra com a instrução correspondente.

`ragcritic run --limit N` grava o resultado do método só com N perguntas; o `bench` seguinte completa o
restante a partir do cache.

### Métodos

- **DynaKRAG**: `fixed_k3`, `fixed_k6`, `fixed_k12`, `ircot`, `self_ask`, `crag`, `adaptive_rag`, `s2g_rag`,
  `corag_s2`, `corag_s3`, `par2_rag`, `dynakrag`, `dynakrag_no_compression`, `dynakrag_no_sufficiency`,
  `dynakrag_uniform`, `dynakrag_cap1`, `dynakrag_cap2`, `dynakrag_cap3`.
- **RAG-Critic**: `standard_rag`, `self_refine`, `flare`, `metarag`, `ragcritic`, `ragcritic_n2`,
  `ragcritic_zeroshot_critic`, `ragcritic_rule_planning`, e `errorbench` para o RAG-Error Bench.

Em `--methods`, o nome pode vir com o artigo na frente (`ragcritic/ragcritic`).

### Configuração

Todas as chaves estão comentadas em `ragbench/configs/base.yaml`; os perfis ficam em
`ragbench/configs/profiles/`. Qualquer chave pode ser alterada na linha de comando:

```bash
--set models.generator=qwen2.5:7b
--set models.planner=qwen2.5:7b
--set retrieval.dynakrag.kind=bm25
--set llm.backend=openai --set llm.base_url=http://servidor:8000
```

A alteração de modelo deve ser repetida em todos os comandos de `doctor` e `bench` da mesma rodada. Resultados
de modelos diferentes não devem compartilhar a mesma pasta `ragbench_results/<perfil>`: a execução é retomável
e trataria as respostas existentes como concluídas.

Arquivos oficiais de datasets já baixados podem ser apontados em `data.local`.

## 3. Painel

`python -m ragbench ui` abre `http://127.0.0.1:8765`.

| Aba | Conteúdo |
|---|---|
| 1 · Testagem | seleção de datasets, artigos e tiers; etapas pendentes e tempo estimado; execução e saída ao vivo; tabela de resultados |
| 2 · Falhas | por método, quantos testes falharam e em que etapa; mapa de falhas; lista de testes com filtros e exportação em CSV |
| 3 · Falhas em comum | perguntas em que os métodos escolhidos falham juntos; sobreposição par a par; divisão por tipo de pergunta |
| Dados | perguntas e gabaritos, busca no corpus, RAG-Error Bench exemplo por exemplo |
| Relatórios | os arquivos `.md` gerados |

Um teste é uma pergunta respondida por um método. Cada teste abre a pergunta, o gabarito, os parágrafos de
apoio e, para cada método, a resposta, os documentos usados e o passo a passo.

O painel atende apenas a própria máquina (127.0.0.1), executa somente comandos do `ragbench` e roda uma rodada
por vez. Sem Ollama ele abre normalmente, para consulta dos resultados existentes.

### Execução em máquina remota

O painel não é exposto na rede. Para acessá-lo a partir de outra máquina, usar um túnel SSH:

```bash
ssh -L 8765:localhost:8765 usuario@maquina
python -m ragbench ui --no-browser        # na máquina remota
```

e abrir `http://localhost:8765` no navegador local. Em rodadas longas por SSH, recomenda-se `tmux` ou
equivalente. Para trazer os resultados, copiar `ragbench_results/` ou exportar o painel com `ui --export`.

## 4. Análise de falhas

Esta análise não consta nos artigos; é um recurso de apoio do pacote.

**Critério de falha.** Um teste falha quando EM = 0. O RAG-Critic e o DynaKRAG reportam apenas médias; a
definição por pergunta segue Jiao et al. (*Doctor-RAG*, arXiv:2604.00865). O painel oferece dois critérios
alternativos (F1 abaixo de um corte; gabarito ausente da resposta), configuráveis em `compare.fail_metric`.

**Etapa da falha**, decidida com o gabarito, nesta ordem:

| Etapa | Quando |
|---|---|
| formato | o gabarito aparece dentro da resposta, ou a resposta é um trecho do gabarito |
| busca | algum parágrafo de apoio não foi recuperado |
| raciocínio | todos os parágrafos de apoio foram recuperados e a resposta está errada |
| indefinido | o dataset não anota parágrafos de apoio e a resposta não é localizável por texto |

**Quebras e consertos.** Cada método é comparado com a sua base nas mesmas perguntas (Standard RAG para o
RAG-Critic; o melhor Fixed-K para o DynaKRAG). Quebra: a base acerta e o método falha. Conserto: o inverso.

**Falhas em comum.** Só são cruzados os datasets usados pelos dois artigos (HotpotQA e 2Wiki), pois apenas
neles os métodos respondem às mesmas perguntas.

### Diagnóstico das falhas de busca e de raciocínio

Este teste não consta nos artigos. Cada pergunta em que o RAG-Critic ou o DynaKRAG falhou por **busca** ou
por **raciocínio** é respondida de novo pela etapa de geração do próprio método, com os documentos já prontos:

| Condição | Documentos entregues ao método |
|---|---|
| `suporte` | só os parágrafos de suporte do gabarito |
| `suporte+busca` | os documentos que o método recuperou na rodada, com os parágrafos de suporte que faltavam no lugar dos últimos documentos que não são de suporte (o total não muda; posição sorteada com semente fixa por pergunta) |

No RAG-Critic, a geração é a do gerador π_α com o Prompt A (a resposta final do método quando o crítico não
acusa erro). No DynaKRAG, é a etapa final do método: compressão da evidência e resposta.

```bash
python -m ragbench diagnostico --profile mini                       # todas as falhas de busca e raciocínio
python -m ragbench diagnostico --profile mini --metodo dynakrag --alvos raciocinio
python -m ragbench diagnostico --profile mini --metodo dynakrag --dataset musique --id <id>   # uma pergunta, em detalhe
python -m ragbench diagnostico --profile mini --criterio f1 --corte 0.5
python -m ragbench diagnostico --profile mini --criterio lev          # Levenshtein normalizado ≥ 0,7
```

Com `--id`, o terminal mostra os documentos de cada condição (★ marca os de suporte), a evidência comprimida do
DynaKRAG, a saída do modelo, a resposta e as métricas. O id de cada pergunta está em `falhas_por_teste.csv`.

**Métricas** (sobre o texto normalizado, com o máximo entre as respostas de referência): EM e F1 do avaliador do
SQuAD/HotpotQA, as métricas dos dois artigos, e duas métricas adicionais: o BLEU (Papineni et al., 2002), calculado
como o `sentence_bleu` do sacreBLEU (add-k com k = 1, ordem efetiva), e a similaridade de Levenshtein normalizada,
Lev = 1 − d / max(|resposta|, |gabarito|), com d = distância de Levenshtein em caracteres (inserção, remoção e
substituição, custo 1). Lev = 1 quando os textos normalizados são iguais e 0 quando não há nada em comum. A
implementação coincide com a `Levenshtein.normalized_similarity` do rapidfuzz (0 divergências em 5.000 pares
aleatórios).

**Critério de acerto.** Padrão: BLEU ≥ 0,5 (`--criterio bleu --corte 0.5`). Alternativas: `--criterio em` ou
`--criterio f1` (corte 0,5) ou `--criterio lev` (corte 0,7), com outro corte por `--corte`. O BLEU mede palavras e sequências em comum, não correção, e não existe um
corte padrão para respostas curtas. O corte foi calibrado contra julgamento manual dos 161 pares resposta-gabarito
distintos do diagnóstico do perfil `mini` (qwen2.5:7b): cortes de 0,40 a 0,50 concordaram com o julgamento em
145 dos 161 pares (90%); o EM = 1, em 141 (88%). Na faixa intermediária, o BLEU erra nos dois sentidos: dá 0,71
a "Felix Körling" para o gabarito "August Körling" (errada) e 0,00 a "Kairi" para "Kairi in the video game series
Kingdom Hearts" (certa). Nos mesmos 161 pares, o Lev concordou com o julgamento em 147 (91,3%) com corte 0,7
(1 falso acerto, 13 falsos erros) e em 141 (87,6%) com corte 0,5 (10 e 10); o Lev também é 0 para "3" e "three".
O julgamento manual não foi revisado por uma segunda pessoa.

**Rótulo de cada pergunta:**

| `suporte` | `suporte+busca` | Rótulo |
|---|---|---|
| acerta | acerta | resolve com os documentos certos |
| acerta | erra | os demais documentos atrapalham |
| erra | acerta | inconsistente |
| erra | erra | erra mesmo com os documentos certos |

Saídas em `ragbench_results/<perfil>/diagnostico/`: `DIAGNOSTICO.md` (contagem dos rótulos, médias das
métricas por condição e a lista de perguntas), `diagnostico.csv` e `diagnostico.jsonl` (saída completa do modelo
em cada condição). As chamadas ao modelo ficam no cache: repetir o comando não gera custo.

### Baseline LLM base

O mesmo gerador respondendo às mesmas perguntas sem nenhum método e sem recuperação (só a pergunta, com o prompt
de resposta sem recuperação do pacote, o mesmo da rota "sem retrieval" do Adaptive-RAG). **Não consta nos
artigos**: é a referência comum pedida pelo orientador para comparar o RAG-Critic e o DynaKRAG.

```bash
python -m ragbench llm-base --profile mini                  # datasets dos dois artigos, 30 perguntas cada
python -m ragbench llm-base --profile mini --datasets hotpotqa musique
```

Saídas em `ragbench_results/<perfil>/llm_base/`: `<dataset>.jsonl` (uma linha por pergunta), `LLM_BASE.md` e
`llm_base.csv`. O relatório põe o LLM base ao lado das rodadas já gravadas (Standard RAG, RAG-Critic, melhor
Fixed-K e DynaKRAG), restritas às mesmas perguntas, com EM, F1, BLEU, Lev e tokens por pergunta. Rodar depois do
`bench`; repetir o comando só refaz o relatório (as respostas ficam no cache).

## 5. Desempenho

- **Índices na GPU.** Com `ollama pull bge-large`, o `prepare` monta os índices do BGE pelo Ollama e confere,
  em uma amostra, que os vetores coincidem com os da CPU. Para forçar a CPU: `--set retrieval.embed_backend=st`.
- **Perguntas em paralelo.** Definir `OLLAMA_NUM_PARALLEL=2`, reiniciar o Ollama e usar `--set llm.workers=2`
  (ou `.\run_all.ps1 -Workers 2`). Conferir no `doctor` que o modelo continua inteiro na GPU.

## 6. Solução de problemas

| Sintoma | Ação |
|---|---|
| `painel.bat` abre e fecha | executar `.\run_all.ps1 -Painel` em um PowerShell aberto na pasta, para ler a mensagem |
| "token inválido — recarregue a página" | o painel foi reiniciado com a página aberta: recarregar |
| porta 8765 ocupada | o painel usa a seguinte e mostra o endereço; ou `--port 9000` |
| o `doctor` indica que o modelo não está inteiro na GPU | fechar outros programas que usam a GPU; se persistir, `--set llm.num_ctx=4096` |
| `prepare` falha em um dataset | os demais seguem; apontar o arquivo oficial em `data.local` ou restringir com `--datasets` |
| erro ao baixar o modelo de embeddings | `--set retrieval.dynakrag.kind=bm25` (e o equivalente em `retrieval.ragcritic.kind`) |
| FLARE informa "no-logprobs" | atualizar o Ollama; sem logprobs o FLARE busca sempre, o que fica registrado no relatório |
| a rodada parou | repetir o mesmo comando: ela continua de onde parou |
| recomeçar do zero | apagar `ragbench_results/<perfil>` e, se desejado, `ragbench_data/llm_cache.sqlite` |
