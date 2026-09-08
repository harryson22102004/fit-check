"""Canonical Hugging Face dataset registry for this lab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    hub_id: str
    kind: str  # asr | mt | st
    language: str
    config: str | None
    splits: tuple[str, ...]
    audio: str | None
    source_text: str
    target_text: str | None
    license: str
    why: str


DATASETS: dict[str, DatasetSpec] = {
    "kokborok_asr": DatasetSpec(
        hub_id="sulabhkatiyar/ne-asr-dataset-trp",
        kind="asr",
        language="trp",
        config=None,
        splits=("train", "validation", "test"),
        audio="audio",
        source_text="text",
        target_text=None,
        license="CC-BY-4.0 (derived from ARTPARK-IISc Vaani)",
        why="3,064 Kokborok clips, 16 kHz mono WAV, romanized transcripts. Start here.",
    ),
    "kokborok_asr_aug": DatasetSpec(
        hub_id="sulabhkatiyar/ne-asr-dataset-trp-aug",
        kind="asr",
        language="trp",
        config=None,
        splits=("train", "validation", "test"),
        audio="audio",
        source_text="text",
        target_text=None,
        license="CC-BY-4.0",
        why="Speed-only 3x augmentation. Do not use pitch-shifted v1.",
    ),
    "kokborok_mt": DatasetSpec(
        hub_id="sdmy/kokborok",
        kind="mt",
        language="trp",
        config=None,
        splits=("train",),
        audio=None,
        source_text="kokborok",
        target_text="english",
        license="MIT (SMOL-derived, locally corrected)",
        why="Kokborok ↔ English parallel text with dialectal corrections.",
    ),
    "kokborok_adivaani": DatasetSpec(
        hub_id="Adivaani/tribal-english-parallel",
        kind="mt",
        language="trp",
        config="trp",
        splits=("train",),
        audio=None,
        source_text="tribal_text",
        target_text="eng_text",
        license="see hub card",
        why="Tribal-English parallel; Kokborok config trp if the hub name resolves.",
    ),
    "bengali_banspeech": DatasetSpec(
        hub_id="SUST-CSE-Speech/banspeech",
        kind="asr",
        language="bn",
        config=None,
        splits=("sports",),
        audio="audio",
        source_text="text",
        target_text=None,
        license="see hub card",
        why="Spontaneous / multi-domain Bengali ASR including dialectal sets.",
    ),
    "bengali_indicvoices_st": DatasetSpec(
        hub_id="ai4bharat/IndicVoices-ST",
        kind="st",
        language="bn",
        config="indic2en",
        splits=("bengali",),
        audio="audio",
        source_text="text",
        target_text="en_text",
        license="CC-BY-4.0",
        why="BhasaAnuvaad Indic→English speech translation (Bengali split).",
    ),
    "marathi_indicvoices_st": DatasetSpec(
        hub_id="ai4bharat/IndicVoices-ST",
        kind="st",
        language="mr",
        config="indic2en",
        splits=("marathi",),
        audio="audio",
        source_text="text",
        target_text="en_text",
        license="CC-BY-4.0",
        why="BhasaAnuvaad Indic→English speech translation (Marathi split).",
    ),
    "wordproject": DatasetSpec(
        hub_id="ai4bharat/NPTEL",
        kind="st",
        language="multi",
        config="indic2en",
        splits=("train",),
        audio="audio",
        source_text="text",
        target_text="en_text",
        license="see hub card",
        why="Investigate NPTEL + WordProject BhasaAnuvaad subsets for bn/mr.",
    ),
}

PRETRAINED = {
    "kokborok_asr": "sulabhkatiyar/ne-asr-trp",
    "mms_base": "facebook/mms-1b-all",
    "indic_whisper": "vasista22/whisper-hindi-large-v2",
    "indic_seamless": "ai4bharat/indic-seamless",
    "nllb": "facebook/nllb-200-distilled-600M",
    "whisper_tiny": "openai/whisper-tiny",
}
