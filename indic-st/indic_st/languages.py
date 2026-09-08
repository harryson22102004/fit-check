"""Language facts and transfer recipe for low-resource Indic ST."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    name: str
    iso639_3: str
    iso639_1: str | None
    family: str
    nearest: tuple[str, ...]
    tonal: bool
    asr_script: str
    transfer_from: tuple[str, ...]
    notes: str


KOKBOROK = Language(
    name="Kokborok",
    iso639_3="trp",
    iso639_1=None,
    family="Sino-Tibetan → Tibeto-Burman → Bodo–Garo",
    nearest=("Bodo (brx)", "Dimasa", "Garo", "Tiwa"),
    tonal=True,
    asr_script="Latin / romanized (NE-ASR); Bengali script in some community text",
    transfer_from=(
        "facebook/mms-1b-all + sulabhkatiyar/ne-asr-trp adapter",
        "Bodo ASR if available (genetic neighbour)",
        "IndicWhisper Bengali only as areal/contact prior, not family prior",
    ),
    notes="Pitch-shift augmentation destroys H/L tone. Speed perturbation only (0.9 / 1.0 / 1.1).",
)

BENGALI = Language(
    name="Bengali",
    iso639_3="ben",
    iso639_1="bn",
    family="Indo-European → Indo-Aryan → Eastern",
    nearest=("Assamese", "Sylheti", "Odia"),
    tonal=False,
    asr_script="Bengali",
    transfer_from=("ai4bharat/indic-seamless", "IndicWhisper bn", "BhasaAnuvaad indic2en"),
    notes="High-resource relative to Kokborok. Use for cascade ST and as a teacher in multilingual batches.",
)

MARATHI = Language(
    name="Marathi",
    iso639_3="mar",
    iso639_1="mr",
    family="Indo-European → Indo-Aryan → Southern",
    nearest=("Hindi", "Konkani"),
    tonal=False,
    asr_script="Devanagari",
    transfer_from=("ai4bharat/indic-seamless", "IndicWhisper mr", "BhasaAnuvaad indic2en"),
    notes="BhasaAnuvaad WordProject / NPTEL subsets are the first ST data sources.",
)

LANGUAGES = {"trp": KOKBOROK, "bn": BENGALI, "mr": MARATHI, "ben": BENGALI, "mar": MARATHI}


def describe(code: str) -> Language:
    key = code.lower()
    if key not in LANGUAGES:
        raise KeyError(f"Unknown language {code}. Use trp, bn, or mr.")
    return LANGUAGES[key]
