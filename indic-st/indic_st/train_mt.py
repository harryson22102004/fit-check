"""Fine-tune a seq2seq MT model: Kokborok / Bengali / Marathi → English."""

from __future__ import annotations

import argparse

from indic_st.registry import DATASETS, PRETRAINED


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="kokborok_mt")
    p.add_argument("--init-from", default=PRETRAINED["nllb"])
    p.add_argument("--output-dir", default="checkpoints/kokborok-nllb")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    spec = DATASETS[args.dataset]
    load_kw = {"path": spec.hub_id}
    if spec.config:
        load_kw["name"] = spec.config
    raw = load_dataset(**load_kw)
    split = "train" if "train" in raw else list(raw.keys())[0]
    ds = raw[split]
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.init_from)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.init_from)

    src_col = spec.source_text
    tgt_col = spec.target_text or "english"

    def guess(example):
        if src_col not in example:
            for k in example:
                if k != tgt_col and isinstance(example[k], str):
                    return k
        return src_col

    sample = ds[0]
    src_key = src_col if src_col in sample else guess(sample)
    tgt_key = tgt_col if tgt_col in sample else next(
        k for k in sample if k != src_key and isinstance(sample[k], str)
    )

    def prep(batch):
        model_in = tok(batch[src_key], max_length=128, truncation=True)
        with tok.as_target_tokenizer():
            labels = tok(batch[tgt_key], max_length=128, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    cols = ds.column_names
    ds = ds.map(prep, batched=True, remove_columns=cols)
    collator = DataCollatorForSeq2Seq(tok, model=model)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=5e-5,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=200,
        report_to=[],
        predict_with_generate=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
