# Avisos de terceiros

## RAG-Critic (repositório oficial)

`ragbench/ragcritic/prompts.py` reproduz textos do repositório oficial do RAG-Critic
(<https://github.com/dongguanting/RAG-Critic>, licença MIT): o prompt do planner
(`critic_agent/prompt_template.py`), os prompts das funções do executor
(`critic_agent/agent_prompt_template.py`) e o formato dos prompts do RAG-Error Bench
(`rag_error_bench/test_data`). As métricas de `ragbench/ragcritic/errorbench.py` são um porte de
`rag_error_bench/caculate_acc.py`. Os arquivos do benchmark são baixados em tempo de execução e não estão
incluídos neste repositório.

```
MIT License

Copyright (c) 2025 KABI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Modelos e dados

Baixados em tempo de execução, não incluídos neste repositório: Qwen2.5 (Apache 2.0), RAG-Critic-3B
(conversão GGUF por mradermacher), BGE-large-en-v1.5 (MIT), E5-base-v2 (MIT), HotpotQA (CC BY-SA 4.0),
2WikiMultihopQA (Apache 2.0), MuSiQue (CC BY 4.0), Natural Questions (CC BY-SA 3.0) e TriviaQA (Apache 2.0).

## Artigos de referência

- Dong et al., *RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval Augmented
  Generation*, ACL 2025.
- Wu et al., *DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented
  Generation*, arXiv:2607.06507, 2026.
- Jiao et al., *Doctor-RAG: A Failure-Aware Repair Framework for Agentic Retrieval-Augmented Generation*,
  arXiv:2604.00865, 2026. Deste artigo vêm apenas a definição de falha por pergunta (EM = 0) e a divisão das
  falhas em formato, busca e raciocínio, usadas na análise de falhas.
