# Avisos de terceiros

## RAG-Critic (RUC-NLPIR)

O formato de entrada do crítico em `dynacritic/prompts.py` (`critic_prompt`) segue o formato
com que o modelo RAG-Critic-3B foi treinado, tal como aparece no repositório
<https://github.com/RUC-NLPIR/RAG-Critic>, distribuído sob a licença MIT:

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

Modelos usados em tempo de execução (baixados por você, não incluídos aqui): RAG-Critic-3B
(conversão GGUF por mradermacher), Qwen2.5 (Apache 2.0), BGE (MIT). Dataset: HotpotQA (CC BY-SA 4.0).

## Artigos de referência

- Wu et al., *DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop
  Retrieval-Augmented Generation*, arXiv:2607.06507, 2026.
- Dong et al., *RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval
  Augmented Generation*, ACL 2025.
