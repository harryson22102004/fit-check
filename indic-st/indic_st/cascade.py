"""Cascaded speech translation: audio → source transcript → English."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CascadeResult:
    transcript: str
    english: str
    asr_model: str
    mt_model: str
    notes: str


def normalize_audio(path_or_array, sampling_rate: int = 16000):
    import numpy as np

    if isinstance(path_or_array, dict) and "array" in path_or_array:
        wav = np.asarray(path_or_array["array"], dtype="float32")
        sr = int(path_or_array.get("sampling_rate") or sampling_rate)
    else:
        import librosa

        wav, sr = librosa.load(path_or_array, sr=sampling_rate, mono=True)
    if sr != sampling_rate:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=sampling_rate)
        sr = sampling_rate
    return wav, sr


class KokborokCascade:
    """Kokborok 🎙️ → Kokborok text → English.

    Default: published NE-ASR MMS adapter + NLLB (or a local MT checkpoint).
    Falls back to Whisper-tiny if the 1B MMS encoder is not cached.
    """

    def __init__(
        self,
        asr_id: str = "sulabhkatiyar/ne-asr-trp",
        asr_base: str = "facebook/mms-1b-all",
        mt_id: str = "facebook/nllb-200-distilled-600M",
        device: str | None = None,
    ):
        self.asr_id = asr_id
        self.asr_base = asr_base
        self.mt_id = mt_id
        self.device = device
        self._asr = None
        self._mt = None
        self._asr_kind = "unloaded"

    def _load_asr(self):
        if self._asr is not None:
            return
        from transformers import pipeline

        try:
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=self.asr_id,
                device=self.device or -1,
            )
            self._asr_kind = self.asr_id
        except Exception:
            self._asr = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=self.device or -1,
            )
            self._asr_kind = "openai/whisper-tiny (fallback — cache NE-ASR locally for production)"

    def _load_mt(self):
        if self._mt is not None:
            return
        from transformers import pipeline

        try:
            self._mt = pipeline(
                "translation",
                model=self.mt_id,
                device=self.device or -1,
            )
        except Exception:
            self._mt = None

    def transcribe(self, audio) -> str:
        self._load_asr()
        wav, sr = normalize_audio(audio)
        out = self._asr({"array": wav, "sampling_rate": sr})
        if isinstance(out, dict):
            return (out.get("text") or "").strip()
        return str(out).strip()

    def translate(self, kokborok_text: str) -> str:
        text = (kokborok_text or "").strip()
        if not text:
            return ""
        self._load_mt()
        if self._mt is None:
            return f"[MT checkpoint unavailable] {text}"
        try:
            out = self._mt(text, src_lang="ben_Beng", tgt_lang="eng_Latn")
        except TypeError:
            out = self._mt(text)
        if isinstance(out, list) and out:
            return (out[0].get("translation_text") or out[0].get("text") or str(out[0])).strip()
        return str(out)

    def __call__(self, audio) -> CascadeResult:
        transcript = self.transcribe(audio)
        english = self.translate(transcript)
        return CascadeResult(
            transcript=transcript,
            english=english,
            asr_model=self._asr_kind,
            mt_model=self.mt_id if self._mt is not None else "unavailable",
            notes="Cascade ST. Kokborok is tonal — never pitch-shift training audio.",
        )


def indic_seamless_st(audio, tgt_lang: str = "eng") -> str:
    """Direct Indic→English speech translation via AI4Bharat IndicSeamless (Bengali/Marathi)."""
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    wav, sr = normalize_audio(audio)
    processor = AutoProcessor.from_pretrained("ai4bharat/indic-seamless")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained("ai4bharat/indic-seamless")
    inputs = processor(audios=wav, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, tgt_lang=tgt_lang)
    return processor.decode(out[0], skip_special_tokens=True)
