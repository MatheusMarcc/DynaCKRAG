"""DPO coarse-to-fine do crítico (Seção 3.2), com TRL ≥ 1.0.

L_DPO(π_θ; π_ref) = −E[log σ(β log π_θ(y+|x)/π_ref(y+|x) − β log π_θ(y−|x)/π_ref(y−|x))],
com π_ref = o modelo SFT (fixo). Hiperparâmetros do artigo: lr 5e-7, scheduler cosseno, warmup 0.1,
batch global 64, perda sigmoid, β = 0.3, 2 épocas, contexto 4096.

Uso:
  python -m ragbench.ragcritic.training.train_dpo --data ragbench_data/critic_training/dpo_official.jsonl \
      --model ragbench_data/critic_training/sft-0.5b --out ragbench_data/critic_training/dpo-0.5b --qlora
"""
from __future__ import annotations

import argparse
import dataclasses
import json


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DPO coarse-to-fine do crítico RAG-Critic (TRL)")
    ap.add_argument("--data", required=True, help="jsonl com {prompt, chosen, rejected} (build-train-data)")
    ap.add_argument("--model", required=True, help="checkpoint do SFT (π_ref inicial)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--lora-lr", type=float, default=5e-6)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args(argv)


def load_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
                if limit and len(rows) >= limit:
                    break
    return rows


def main(argv=None):
    args = parse_args(argv)
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from .train_sft import config_kwargs, model_kwargs, peft_config

    ds = Dataset.from_list(load_rows(args.data, args.max_samples))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs(args))
    use_lora = args.lora or args.qlora
    if args.qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cuda = torch.cuda.is_available() and not args.cpu
    bf16 = cuda and torch.cuda.is_bf16_supported()
    # TRL ≥ 1.0 espera loss_type como lista; versões antigas, como string
    lt_field = next((f for f in dataclasses.fields(DPOConfig) if f.name == "loss_type"), None)
    loss_type = ["sigmoid"] if lt_field is not None and "list" in str(lt_field.type) else "sigmoid"
    cfg = DPOConfig(**config_kwargs(
        DPOConfig, output_dir=args.out, learning_rate=args.lora_lr if use_lora else args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=max(1, args.global_batch // max(1, args.per_device_batch)),
        lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio, beta=args.beta, loss_type=loss_type,
        max_length=args.max_length, logging_steps=10, save_steps=args.save_steps, save_total_limit=2,
        bf16=bf16, fp16=cuda and not bf16, gradient_checkpointing=cuda, seed=args.seed, report_to=[],
        use_cpu=not cuda,
    ))
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok,
                         peft_config=peft_config() if use_lora else None)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"modelo DPO salvo em {args.out}")


if __name__ == "__main__":
    main()
