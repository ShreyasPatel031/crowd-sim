#!/usr/bin/env python3
"""QLoRA SFT of Qwen2.5-Instruct on converted OPeRA next-action examples.

Loss is next-token SFT on the assistant action JSON only (completion-only).
This is the practical stand-in for the paper's 64×H200 FSDP run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opera_repro.config import load_config
from opera_repro.converter import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--train", default=str(ROOT / "data" / "processed" / "train.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "data" / "processed" / "val.jsonl"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    output_dir = args.output_dir or train_cfg["output_dir"]

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training extras are missing. Create the venv and install:\n"
            "  python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt"
        ) from exc

    train_rows = _load_messages(args.train)
    val_rows = _load_messages(args.val) if Path(args.val).exists() else None
    if not train_rows:
        raise SystemExit(f"No training rows in {args.train}. Run scripts/prepare_data.py first.")

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if use_cuda:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["torch_dtype"] = torch.bfloat16
        print("Loading 4-bit QLoRA base on CUDA")
    else:
        model_kwargs["torch_dtype"] = torch.float32
        print("CUDA not found — loading full-precision LoRA (slow; fine for a tiny smoke run)")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    peft_config = LoraConfig(
        r=int(train_cfg["lora_r"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    sft_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg["epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        lr_scheduler_type="cosine",
        bf16=use_cuda,
        gradient_checkpointing=True,
        report_to="none",
    )
    max_len = int(train_cfg.get("max_seq_len", cfg["model"]["max_seq_len"]))
    sft_config = _build_sft_config(SFTConfig, sft_kwargs, max_len)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows) if val_rows else None,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    meta = {"base_model": model_name, "train_rows": len(train_rows), "output_dir": output_dir}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved adapter to {output_dir}")


def _build_sft_config(SFTConfig, kwargs, max_len):
    extra_attempts = [
        {"max_length": max_len, "assistant_only_loss": True},
        {"max_seq_length": max_len, "assistant_only_loss": True},
        {"max_length": max_len},
        {"max_seq_length": max_len},
        {},
    ]
    last_error = None
    for extra in extra_attempts:
        try:
            return SFTConfig(**kwargs, **extra)
        except TypeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return SFTConfig(**kwargs)


def _load_messages(path: str) -> list[dict]:
    return [{"messages": row["messages"]} for row in iter_jsonl(path)]


if __name__ == "__main__":
    main()
