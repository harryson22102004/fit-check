#!/usr/bin/env bash
set -euo pipefail
python -m indic_st.train_asr --dataset kokborok_asr --init-from openai/whisper-tiny --max-samples 32 --max-steps 20 --output-dir checkpoints/smoke-asr
python -m indic_st.train_mt --dataset kokborok_mt --max-samples 32 --max-steps 20 --output-dir checkpoints/smoke-mt
