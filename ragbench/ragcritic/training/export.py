"""Exporta o crítico treinado para o Ollama: funde o LoRA (se houver) e escreve o Modelfile.

Passos (o de conversão para GGUF usa o llama.cpp, que não é dependência deste projeto):
  1. python -m ragbench.ragcritic.training.export --adapter <saida_dpo> --base <modelo_sft_ou_base> --out <pasta_merged>
  2. python llama.cpp/convert_hf_to_gguf.py <pasta_merged> --outfile critic.gguf --outtype q8_0
  3. ollama create critico-local -f <pasta_merged>/Modelfile
  4. python -m ragbench ragcritic errorbench --set models.critic=critico-local
"""
from __future__ import annotations

import argparse
import os

MODELFILE = '''FROM ./critic.gguf
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
SYSTEM "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description="funde LoRA e prepara o crítico para o Ollama")
    ap.add_argument("--adapter", required=True, help="pasta do treino (LoRA) ou modelo completo")
    ap.add_argument("--base", default=None, help="modelo base do LoRA (se --adapter for um LoRA)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.path.exists(os.path.join(a.adapter, "adapter_config.json")):
        from peft import PeftModel

        if not a.base:
            raise SystemExit("--base é obrigatório para fundir um LoRA")
        base = AutoModelForCausalLM.from_pretrained(a.base)
        model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
        tok = AutoTokenizer.from_pretrained(a.base)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.adapter)
        tok = AutoTokenizer.from_pretrained(a.adapter)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "Modelfile"), "w", encoding="utf-8") as f:
        f.write(MODELFILE)
    print(f"modelo fundido em {a.out}\nPróximo passo: converter para GGUF com o llama.cpp (ver docstring).")


if __name__ == "__main__":
    main()
