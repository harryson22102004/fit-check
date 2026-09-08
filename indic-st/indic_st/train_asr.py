"""Fine-tune Whisper-family ASR with cross-lingual initialisation.

Kokborok recipe
---------------
1. Preferred published system: MMS-1B frozen encoder + language CTC adapter
   (`sulabhkatiyar/ne-asr-trp`). That is the NE-ASR paper setup.
2. This script fine-tunes a Whisper encoder-decoder instead, starting from
   a *nearest-language* checkpoint (Bodo if present, else Bengali IndicWhisper
   as areal contact, else whisper-small). Use only speed perturbation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from indic_st.languages import KOKBOROK
from indic_st.registry import DATASETS, PRETRAINED


@dataclass
class ASRTrainConfig:
    dataset: str = "kokborok_asr"
    init_from: str = "openai/whisper-small"
    output_dir: str = "checkpoints/kokborok-whisper"
    max_steps: int = 800
    batch_size: int = 8
    lr: float = 1e-5
    max_samples: int | None = None
    language: str = "trp"


NEAREST_INIT = {
    "trp": "openai/whisper-small",  # swap to a Bodo Whisper if you host one
    "bn": "vasista22/whisper-bengali-small",
    "mr": "vasista22/whisper-marathi-small",
}


def parse_args() -> ASRTrainConfig:
    p = argparse.ArgumentParser(description="Fine-tune Whisper ASR for Kokborok / bn / mr")
    p.add_argument("--dataset", default="kokborok_asr", choices=list(DATASETS))
    p.add_argument("--init-from", default=None, help="Pretrained checkpoint (cross-lingual start)")
    p.add_argument("--output-dir", default="checkpoints/kokborok-whisper")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()
    spec = DATASETS[args.dataset]
    init = args.init_from or NEAREST_INIT.get(spec.language, PRETRAINED["whisper_tiny"])
    return ASRTrainConfig(
        dataset=args.dataset,
        init_from=init,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        language=spec.language,
    )


def main() -> None:
    cfg = parse_args()
    print("=== Kokborok / Indic ASR fine-tune ===")
    print(f"Language family note: {KOKBOROK.family}")
    print(f"Nearest languages: {', '.join(KOKBOROK.nearest)}")
    print(f"Tonal: {KOKBOROK.tonal} → {KOKBOROK.notes}")
    print(f"Init (cross-lingual): {cfg.init_from}")
    print(f"Data: {DATASETS[cfg.dataset].hub_id}")

    from datasets import Audio, load_dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    spec = DATASETS[cfg.dataset]
    load_kw = {"path": spec.hub_id}
    if spec.config:
        load_kw["name"] = spec.config
    raw = load_dataset(**load_kw)
    split = "train" if "train" in raw else list(raw.keys())[0]
    ds = raw[split]
    if spec.audio:
        ds = ds.cast_column(spec.audio, Audio(sampling_rate=16000))
    if cfg.max_samples:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))

    processor = WhisperProcessor.from_pretrained(cfg.init_from)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.init_from)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    text_col = spec.source_text if spec.source_text in ds.column_names else "text"
    audio_col = spec.audio or "audio"

    def prep(batch):
        audio = batch[audio_col]
        feats = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
        )
        batch["input_features"] = feats.input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    ds = ds.map(prep, remove_columns=ds.column_names)

    class Collator:
        def __call__(self, features):
            import torch
            from torch.nn.utils.rnn import pad_sequence

            inputs = torch.tensor([f["input_features"] for f in features])
            labels = pad_sequence(
                [torch.tensor(f["labels"]) for f in features],
                batch_first=True,
                padding_value=-100,
            )
            return {"input_features": inputs, "labels": labels}

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        max_steps=cfg.max_steps,
        fp16=False,
        logging_steps=10,
        save_steps=200,
        report_to=[],
        predict_with_generate=True,
        generation_max_length=128,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=Collator(),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"Saved {cfg.output_dir}")


if __name__ == "__main__":
    main()
