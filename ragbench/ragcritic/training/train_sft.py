"""SFT do crítico (Seção 3.2, "Supervised Fine-tuning"), com TRL ≥ 1.0.

Hiperparâmetros do artigo (apêndice B.2): lr 7e-6, 3 épocas, scheduler linear, 20 passos de warmup,
weight decay 0.1, batch global 128, contexto 2048, bf16, DeepSpeed ZeRO-3 em 8×A800.
Para uma GPU de 4 GB: ``--qlora`` com Qwen2.5-0.5B-Instruct, batch 1 + acumulação.

Uso:
  python -m ragbench.ragcritic.training.train_sft --data ragbench_data/critic_training/sft_official.jsonl \
      --model Qwen/Qwen2.5-0.5B-Instruct --out ragbench_data/critic_training/sft-0.5b --qlora --max-samples 4000
"""
from __future__ import annotations

import argparse
import json


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SFT do crítico RAG-Critic (TRL)")
    ap.add_argument("--data", required=True, help="jsonl com {prompt, completion} (build-train-data)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=7e-6)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--global-batch", type=int, default=128)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=4096, help="artigo: 2048; README oficial: 4096 (evita cortar a saída)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--lora", action="store_true", help="LoRA (r=16) em vez de fine-tuning completo")
    ap.add_argument("--qlora", action="store_true", help="LoRA sobre o modelo em 4 bits (bitsandbytes, CUDA)")
    ap.add_argument("--lora-lr", type=float, default=2e-4, help="lr usado com --lora/--qlora")
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cpu", action="store_true", help="força CPU (só para teste)")
    return ap.parse_args(argv)


def load_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"prompt": r["prompt"], "completion": r["completion"]})
                if limit and len(rows) >= limit:
                    break
    return rows


def peft_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def config_kwargs(config_cls, **kw) -> dict:
    """Compatibilidade transformers 4.x/5.x: ``warmup_ratio`` sumiu no 5.x (vira ``warmup_steps`` < 1)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(config_cls)}
    if "warmup_ratio" in kw and "warmup_ratio" not in names:
        kw["warmup_steps"] = kw.pop("warmup_ratio")
    return {k: v for k, v in kw.items() if k in names}


def model_kwargs(args) -> dict:
    import torch

    kw: dict = {}
    cuda = torch.cuda.is_available() and not args.cpu
    if cuda:
        kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if args.qlora:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=kw.get("torch_dtype", torch.float16),
                                                       bnb_4bit_use_double_quant=True)
        kw["device_map"] = "auto"
    return kw


def main(argv=None):
    args = parse_args(argv)
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = load_rows(args.data, args.max_samples)
    ds = Dataset.from_list(rows)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs(args))
    use_lora = args.lora or args.qlora
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    accum = max(1, args.global_batch // max(1, args.per_device_batch))
    cuda = torch.cuda.is_available() and not args.cpu
    bf16 = cuda and torch.cuda.is_bf16_supported()
    cfg = SFTConfig(**config_kwargs(
        SFTConfig, output_dir=args.out, learning_rate=args.lora_lr if use_lora else args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch, gradient_accumulation_steps=accum,
        lr_scheduler_type="linear", warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        max_length=args.max_length, logging_steps=10, save_steps=args.save_steps, save_total_limit=2,
        bf16=bf16, fp16=cuda and not bf16, gradient_checkpointing=cuda, seed=args.seed, report_to=[],
        use_cpu=not cuda,
    ))
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=peft_config() if use_lora else None)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"modelo SFT salvo em {args.out}")


if __name__ == "__main__":
    main()
