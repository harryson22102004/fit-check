---
language:
  - trp
  - bn
  - mr
  - en
tags:
  - speech-translation
  - automatic-speech-recognition
  - kokborok
  - low-resource
  - indic
license: mit
library_name: transformers
pipeline_tag: automatic-speech-recognition
---

# Tripura ST Lab

Training code and dataset registry for **Kokborok-first speech translation**, then Bengali and Marathi on BhasaAnuvaad.

Landing page: https://huggingface.co/spaces/vivekharry/tripura-st-lab-site

```
Kokborok AUDIO (16 kHz mono)
    → ASR (MMS-1B + NE-ASR adapter, or Whisper fine-tune)
Kokborok transcript
    → MT (NLLB / sdmy/kokborok)
English
```

Clone this repo, then:

```bash
pip install -e .
python -m indic_st.train_asr --dataset kokborok_asr --init-from openai/whisper-small
python -m indic_st.train_mt --dataset kokborok_mt
```

See the full README in the files tab for dataset IDs, tonal-augmentation rules, and citations.

Author: Vivek Das (`vivekharry`).
