"""QLoRA fine-tune a small chat model on YOUR messages. Free-T4 friendly.

    python -m src.train --config configs/qwen3b_t4.yaml

What it does: loads the base model in 4-bit (NF4), formats each example with the
model's chat template, attaches a LoRA adapter to the attention + MLP
projections, and trains with TRL's SFTTrainer. Two quality features:

  * completion-only loss  -> gradients flow ONLY through your (assistant) tokens,
    so the model learns your voice, not the other person's.
  * eval on a held-out val set (if present) -> watch for overfitting.

Saves just the adapter (a few MB) to `output_dir`. T4 note: Turing GPUs have no
bf16, so this auto-selects fp16. See the config.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/qwen3b_t4.yaml")
    # Lets the notebook override key knobs without editing the yaml.
    ap.add_argument("--model_id")
    ap.add_argument("--data")
    ap.add_argument("--output_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for k in ("model_id", "data", "output_dir"):
        if getattr(args, k):
            cfg[k] = getattr(args, k)

    # fp16 on T4; auto-upgrade to bf16 on capable GPUs (A100/L4/3090+).
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- datasets: render each {"messages": [...]} into one templated string ----
    def render(example: dict) -> dict:
        return {"text": tok.apply_chat_template(example["messages"], tokenize=False)}

    train_ds = load_dataset("json", data_files=cfg["data"], split="train")
    train_ds = train_ds.map(render, remove_columns=train_ds.column_names)

    eval_ds = None
    val_path = cfg.get("val_data")
    if val_path and Path(val_path).exists():
        eval_ds = load_dataset("json", data_files=val_path, split="train")
        eval_ds = eval_ds.map(render, remove_columns=eval_ds.column_names)

    # ---- 4-bit base model ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.use_cache = False  # required with gradient checkpointing

    # ---- LoRA ----
    peft_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- completion-only collator: train ONLY on your (assistant) tokens ----
    collator = None
    if cfg.get("completion_only", False):
        collator = DataCollatorForCompletionOnlyLM(
            response_template=cfg["response_template"],
            instruction_template=cfg["instruction_template"],
            tokenizer=tok,
        )

    # ---- trainer config (built as a dict so we can add eval conditionally) ----
    sft_kwargs = dict(
        output_dir=cfg["output_dir"],
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        seed=cfg["seed"],
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",  # 8-bit optimizer keeps T4 memory in check
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    if eval_ds is not None:
        # transformers renamed evaluation_strategy -> eval_strategy; pick the
        # one this installed version exposes.
        sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}
        eval_key = "eval_strategy" if "eval_strategy" in sft_fields else "evaluation_strategy"
        sft_kwargs[eval_key] = "epoch"
        sft_kwargs["per_device_eval_batch_size"] = cfg.get(
            "per_device_eval_batch_size", cfg["per_device_train_batch_size"]
        )
    sft_args = SFTConfig(**sft_kwargs)

    # TRL renamed `tokenizer` -> `processing_class` in newer versions; pick
    # whichever this installed version accepts so we work across the churn.
    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_cfg,
        data_collator=collator,
    )
    tok_param = "processing_class" if "processing_class" in inspect.signature(
        SFTTrainer.__init__
    ).parameters else "tokenizer"
    trainer_kwargs[tok_param] = tok
    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"\nSaved LoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
