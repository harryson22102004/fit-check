---
title: Tripura ST Lab
emoji: 🎙️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
pinned: false
license: mit
tags:
  - speech-translation
  - kokborok
  - asr
  - low-resource
  - indic
---

# Tripura ST Lab — Kokborok first, then Bengali and Marathi

Cascaded **speech translation** for a low-resource North-East language, then the same recipe on BhasaAnuvaad-scale Indic ST.

```
Kokborok AUDIO (16 kHz mono WAV)
        ↓  ASR  (MMS-1B frozen + NE-ASR CTC adapter, or Whisper fine-tune)
Kokborok TRANSCRIPT (romanized)
        ↓  MT   (NLLB / local seq2seq on sdmy/kokborok)
English
```

## Why this order

Kokborok (`trp`) is **Tibeto-Burman, Bodo–Garo, tonal**. Nearest labelled neighbours: **Bodo, Dimasa, Garo**. Do **not** pitch-shift. Speed perturbation 0.9 / 1.0 / 1.1 only.

Bengali and Marathi sit on **BhasaAnuvaad** (44k+ hours, Indic→English). They are the high-resource tracks and the place to use **IndicSeamless** / **IndicWhisper** as pretrained teachers.

## Datasets wired in `indic_st/registry.py`

| ID | Hub | Role |
| --- | --- | --- |
| kokborok_asr | `sulabhkatiyar/ne-asr-dataset-trp` | 3064 clips, 2511/274/279, 16 kHz, CC-BY-4.0, Vaani |
| kokborok_asr_aug | `sulabhkatiyar/ne-asr-dataset-trp-aug` | 3× speed-only |
| kokborok_mt | `sdmy/kokborok` | SMOL + local dialect correction, Kokborok↔English |
| kokborok_adivaani | Adivaani tribal-English `trp` | `tribal_text` → `eng_text` |
| bengali_banspeech | `SUST-CSE-Speech/banspeech` | spontaneous Bengali ASR |
| bengali_indicvoices_st | `ai4bharat/IndicVoices-ST` `indic2en` | Bengali ST |
| marathi_indicvoices_st | same collection | Marathi ST |
| wordproject / NPTEL | AI4Bharat BhasaAnuvaad | lecture ST, bn/mr fields |

Published Kokborok ASR checkpoint: `sulabhkatiyar/ne-asr-trp` (adapter on frozen `facebook/mms-1b-all`). Direct Indic ST: `ai4bharat/indic-seamless`.

## Train locally (GPU)

```bash
pip install -e .
huggingface-cli login

python -m indic_st.train_asr --dataset kokborok_asr --init-from openai/whisper-small
python -m indic_st.train_mt  --dataset kokborok_mt  --init-from facebook/nllb-200-distilled-600M
```

Cross-lingual init: start Whisper from a Bodo checkpoint if you have one; otherwise `whisper-small` or IndicWhisper-Bengali as an *areal* prior. Prefer the official MMS adapter for production Kokborok ASR.

## Cite

Vaani / NE-ASR, SMOL, BhasaAnuvaad (Jain et al. 2024), BanSpeech, and the hub cards linked above.

Built by Vivek Das.
