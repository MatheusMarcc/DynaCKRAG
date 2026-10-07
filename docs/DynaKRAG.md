# DynaKRAG — notas de implementação

Wu et al., *DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented
Generation*, arXiv:2607.06507 (2026). O artigo não publica código; tudo abaixo vem do texto e do apêndice.

## Mapa artigo → código (`ragbench/dynakrag/`)

| Artigo | Código | Observação |
|---|---|---|
| Estado s_t: pergunta, docs, fronteira, query, lacuna, suficiência, histórico, custo | `state.EvidenceState` | só quantidades observáveis |
| 7 operações atômicas (Tabela 5) | `env.DynaEnv` | cada ação devolve um NOVO estado (permite rotular ramificando) |
| Camada de validade A(s_t) | `validity.valid_actions` | aplicada antes do modelo |
| Features do apêndice ("Runtime features") | `features.py` | 52 features de valor, 44 de continuação |
| Eq. (1): alvos ΔSR / 1[SR=1] / SR | `training.collect_value_data` | o gold só aparece aqui |
| v̂θ: random forest 300 árvores, folha 8, semente 13 | `training.fit_value_model` | validação cruzada agrupada por pergunta |
| Modelo de continuação | `training.collect_continuation_data` + `fit_continuation_model` | ver decisões abaixo |
| Lógica de parada ("stopping and continuation logic", Tabela 5) | `validity.valid_actions` (`stop_rule`, `max_probes`) | ver a decisão 5 |
| Eq. (2) e Algoritmo 1 | `controller.LearnedController` + `inference.run_episode` | λ = 0 |
| Compressão terminal | `env.DynaEnv.compress` | extração (HotpotQA/2Wiki) e tabela de fatos (MuSiQue) |
| Ablação "w/o learned controller" | `controller.UniformController` | uniforme com semente, horizonte fixo, sem continuação |
| Baselines da Tabela 1 | `baselines/__init__.py` | implementações controladas, mesmo gerador e mesmo prompt de resposta |
| Tabelas 1, 2, 6, 7, 8 e Figs. 3, 4, 5 | `report.py` | com os números do artigo ao lado |

## Configuração (Tabela 4 do artigo × pacote)

| Componente | Artigo | ragbench |
|---|---|---|
| Modelo de resposta | Qwen2.5-7B-Instruct | `qwen2.5:3b` por padrão (configurável em `models.generator`) |
| Retriever denso | BGE-large-en-v1.5, FAISS flat | BGE-large-en-v1.5, produto interno exato (numpy) na CPU |
| Corpus | fullwiki do HotpotQA; corpora dos datasets | união dos parágrafos das perguntas amostradas (ver o README) |
| Decodificação | greedy, temperatura 0 | igual (semente 13) |
| Exemplos de treino | 1.000 por dataset | 60 (`mini`) / 150 (`quick`) / 1.000 (`standard`) |
| Modelos de valor e continuação | RF 300 árvores, folha 8, semente 13 | igual |
| Passos de controle | 4 | 4 |
| Docs por aquisição | 3 | 3 |
| Contexto final | 12 docs | 12 docs |
| λ | 0 | 0 |
| Compressão | 8 trechos/256 tokens (HotpotQA, 2Wiki); tabela de fatos 12/384 (MuSiQue); resposta ≤ 48 tokens | igual |

## Decisões onde o artigo é omisso

1. **Política que gera os estados de treino.** O artigo diz "we execute or simulate the transition offline"
   sem dizer quais estados. O pacote usa trajetórias de uma política uniforme sobre as ações válidas (sem parar
   antes do limite). Em cada estado visitado, **todas** as ações válidas são rotuladas: as de aquisição são
   executadas a partir do mesmo estado (daí o estado imutável).
2. **Sonda de suficiência em uma chamada** (`probe: one_step`): o julgamento e a descrição da lacuna saem do
   mesmo JSON. Quem fixa isso é a Tabela 6 do artigo. Com limite de 1 retrieval, são exatamente 3,000
   chamadas ao LLM por pergunta: sonda + compressão + resposta. Com limite 2, são 3,100 / 3,006 / 3,007,
   embora 74% / 80% / 95% das perguntas façam uma segunda busca; a diferença para 3 é só o `rewrite_query`
   (0,106 / 0,009 / 0,009 na Tabela 7). Logo a sonda é uma chamada só e a `gap_query` não chama o LLM.
   O que fica sem explicação: com limite 3, o artigo tem ≈0,56–0,68 chamada a mais por pergunta do que
   sonda + rewrite + compressão + resposta. O texto não diz de onde vem (uma hipótese é a segunda
   `gap_query` pedir uma nova descrição da lacuna); no pacote, a segunda `gap_query` reaproveita a mesma lacuna.
3. **`gap_query` pode repetir a mesma lacuna.** A Tabela 7 tem mais `gap_query` (1,24–1,54) do que sondas
   (≈1). A regra de validade é "existe lacuna registrada", então cada repetição traz os próximos 3 documentos
   ainda não vistos para a mesma lacuna. A validade também exige que a busca ainda tenha documentos novos,
   para evitar transições vazias.
4. **Detector de entidade-ponte.** Não é especificado. São usados nomes próprios (sequências capitalizadas)
   citados na evidência que ainda não têm documento no estado nem aparecem na pergunta. No artigo, a ação
   quase não é usada (0,02–0,11 por pergunta).
5. **Quando o controlador pode parar.** Pela Eq. (1), parar vale SR(s) e adquirir vale ΔSR: com qualquer
   evidência no estado o argmax quase sempre é parar. O artigo resolve isso fora do modelo de valor: na
   Tabela 5, `stop_answer` é "enabled by the stopping and continuation logic or by a hard cap". O texto não
   escreve a regra, mas as tabelas a fixam:

   | O que o artigo mostra | De onde | O que implica |
   |---|---|---|
   | Sem a sonda, "none selects stop_answer": 4,000 passos e 4,0 retrievals em todo exemplo | Tabela 8 | parar depende do resultado da sonda |
   | 0,998 / 0,999 / 0,977 sondas por pergunta, nunca mais de uma | Tabela 7 | a sonda roda no máximo uma vez por trajetória |
   | Com limite de 1 retrieval todo exemplo roda a sonda (3,000 chamadas ao LLM) | Tabela 6 | a sonda vem antes de qualquer parada |
   | 26% / 20% / 5% param depois de uma busca só; 4% / 1% / 0,6% depois de duas; o resto usa as três | Tabela 6 (1 − *Exhausted*, limites 2 e 3) | quem não sai logo depois da sonda vai até o limite |
   | Passos: 3,437 / 3,598 / 3,890 | Tabela 8 | = 2 × (sai depois da sonda) + 3 × (sai depois de mais uma busca) + 4 × (limite), com as frações acima |

   A regra implementada (`dynakrag.controller.stop_rule: sufficient`, `max_probes: 1`):
   `stop_answer` só entra em A(s_t) depois que a sonda disse "suficiente" (ou quando não sobra ação
   executável); a sonda roda no máximo uma vez. Com a parada válida e no topo do ranking, o **modelo de
   continuação** ainda pode segurá-la ("can suppress stop_answer when additional evidence is predicted to
   help"): se P(continuar ajuda) ≥ τ, o controlador faz a melhor aquisição. Isso reproduz as três primeiras
   linhas da tabela por construção; os percentuais das duas últimas dependem do que a sonda responde.

   O modelo de continuação é "trained on trajectory prefixes to estimate the answer utility of continuing":
   a política de valor é executada sem parada até o limite; em cada prefixo, G responde com a evidência
   daquele momento, e o rótulo é `1[F1(fim) > F1(prefixo)]`. O classificador é uma RF 300/8/13. O limiar τ
   é **calibrado** (`continuation_threshold: auto`) simulando a parada só nos prefixos em que ela é permitida,
   com as probabilidades da validação cruzada; em empate fica o menor τ ("permissive continuation
   thresholds"). Com menos de 10 prefixos assim não há o que calibrar e vale τ = 0,5.

   Cada linha de resultado leva a assinatura das regras (campo `control`) e o controlador treinado tem as
   regras no nome do arquivo; resultado de outra assinatura é guardado e recalculado, nunca misturado.
6. **Ranking inicial (fronteira).** A pergunta é buscada no mesmo índice denso, com profundidade 50.
7. **Baselines.** O artigo também usa "controlled implementations". Os do pacote reproduzem a ideia central de
   cada método com o mesmo gerador, o mesmo retriever e o mesmo prompt de resposta final:
   * Fixed-K: grade K ∈ {3, 6, 12}; a Tabela 1 mostra o melhor K por dataset.
   * CoRAG: s2 e s3; mostra o mais forte no geral.
   * Adaptive-RAG: classificador TF-IDF + regressão logística treinado com rótulos "prata" (a estratégia
     mais simples que acerta, entre sem retrieval, 1 passo e IRCoT; sem acerto, vale o viés do dataset
     multi-hop). O artigo original usa um T5.
   * CRAG: o avaliador T5 e a busca na web viram, respectivamente, o LLM como avaliador e uma nova busca
     no corpus.

## Como ler os resultados

* **Tabela 1**: EM, F1 e tokens por pergunta.
* **Tabela 2**: o efeito de cada componente. A pergunta central do artigo é se *completo > sem controlador
  aprendido*, ou seja, se aprender a ordenar ajuda mais do que só expor as ações.
* **Tabela 7 / Fig. 4**: a composição das ações. No artigo o padrão é retrieve → sufficiency → gap
  (→ gap). **Esta tabela deve ser conferida antes da Tabela 1**: se a composição medida for muito diferente
  da do artigo, o controlador não está se comportando como o do DynaKRAG e a comparação de acerto perde o sentido.
* **Tabela 8**: sem a sonda, o controlador nunca para antes do limite no artigo (e no pacote, por construção).
* **Veredito da sonda**: quantas perguntas a sonda julgou suficientes. No artigo isso é ≈26% / 20% / 5%; um
  modelo pequeno que quase nunca diz "suficiente" leva toda pergunta até o limite e gasta mais tokens.
* **Fig. 3**: no artigo, mais retrieval não é sempre melhor (o MuSiQue piora no limite 3).
* **Diagnóstico do controlador**: MAE por ação, acerto do argmax, AUC e τ da continuação, features mais
  importantes.

## Diferenças em relação ao artigo

* **Prompts.** O artigo não publica nenhum; os da sonda, da compressão, da reescrita e da resposta foram
  escritos para o pacote.
* **Gerador.** O padrão é um modelo de 3B, que faz a sonda, a compressão e a resposta. O gerador do artigo
  (`qwen2.5:7b`) pode ser selecionado em `models.generator`.
* **Corpus reduzido.** No fullwiki a busca estática recupera pouco; no corpus reduzido o Fixed-K já recupera
  quase todos os parágrafos de apoio no HotpotQA, de modo que sobra menos para a aquisição adaptativa ganhar.
* **Deduplicação por título.** É a regra do artigo ("identifier- and normalized-title-based deduplication").
  No corpus reduzido do MuSiQue há títulos com mais de um parágrafo: um parágrafo recuperado impede a entrada
  de outro com o mesmo título. O Fixed-K não deduplica.
* **Fixed-K** entra na Tabela 1 com o melhor K escolhido nas próprias perguntas de teste, como no artigo
  ("the strongest non-gold setting from the completed grid"). Isso favorece o Fixed-K.
* **Compressão.** Linhas do compressor sem texto algum ("- - -") são descartadas; se não sobrar nenhuma, a
  resposta é gerada com os documentos (a regra que já valia para a saída vazia). O relatório conta os casos.
