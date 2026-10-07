"""Alinhamento do crítico (Seção 3.2 do RAG-Critic): SFT e depois DPO coarse-to-fine.

* ``build_data`` — monta os dados de SFT (Prompt D → saída JSON do crítico, com erros e corretos em igual
  número) e os pares de preferência do DPO: para cada x, negativos amostrados do conjunto de CORRETOS
  (distinção grossa certo × errado) e do conjunto de ERROS (distinção fina entre tipos de erro).
* ``train_sft`` / ``train_dpo`` — scripts TRL (≥ 1.0) com os hiperparâmetros do artigo; LoRA/QLoRA
  opcionais para GPUs pequenas.
* ``export`` — funde o LoRA e gera o Modelfile para usar o crítico treinado no Ollama.

Em GPUs de 4 GB só cabe treinar um modelo pequeno (Qwen2.5-0.5B, QLoRA). O 3B do artigo precisa de
GPU de laboratório. Os benchmarks usam os pesos oficiais do RAG-Critic-3B.
"""
