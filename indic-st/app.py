"""Gradio lab: Kokborok cascade + dataset registry + transfer recipe."""

from __future__ import annotations

import gradio as gr

from indic_st.languages import BENGALI, KOKBOROK, MARATHI
from indic_st.metrics import bleu1, wer
from indic_st.registry import DATASETS, PRETRAINED

CASCADE = None


def get_cascade():
    global CASCADE
    if CASCADE is None:
        from indic_st.cascade import KokborokCascade

        CASCADE = KokborokCascade()
    return CASCADE


def run_cascade(audio):
    if audio is None:
        return "Record or upload 16 kHz mono audio.", "", "—"
    try:
        result = get_cascade()(audio)
        return result.transcript, result.english, f"{result.asr_model} → {result.mt_model}\n{result.notes}"
    except Exception as exc:
        return (
            "",
            "",
            f"Could not load heavy ASR weights on this Space CPU ({exc}). "
            "Train locally: python -m indic_st.train_asr --dataset kokborok_asr --max-samples 64 --max-steps 50",
        )


def peek_dataset(name: str, n: int):
    try:
        from indic_st.data import iter_pairs

        rows = list(iter_pairs(name, limit=int(n)))
    except Exception as exc:
        spec = DATASETS[name]
        return (
            f"Could not stream `{spec.hub_id}` from this machine:\n\n{exc}\n\n"
            f"Why it matters: {spec.why}\nLicense: {spec.license}"
        )
    lines = [f"**{DATASETS[name].hub_id}** — {DATASETS[name].why}", ""]
    for i, row in enumerate(rows, 1):
        src = (row.get("source") or "")[:240]
        tgt = (row.get("target") or "")[:240]
        keys = ", ".join(row.get("raw_keys") or [])
        lines.append(f"{i}. source: {src or '—'}")
        if tgt:
            lines.append(f"   english: {tgt}")
        lines.append(f"   columns: `{keys}`")
        lines.append("")
    return "\n".join(lines)


def score(ref, hyp):
    return f"WER {wer(ref, hyp):.3f} · unigram overlap {bleu1(ref, hyp):.3f}"


LANG_MD = f"""
### 1. Kokborok first (trp)

- Family: **{KOKBOROK.family}**
- Nearest languages: **{', '.join(KOKBOROK.nearest)}**
- Tonal: **yes** — speed perturbation only (0.9/1.0/1.1). Never pitch-shift.
- ASR labels: romanized Latin (`sulabhkatiyar/ne-asr-dataset-trp`, 2511/274/279, 16 kHz WAV, CC-BY-4.0, Vaani-derived)
- Pretrained adapter: `{PRETRAINED['kokborok_asr']}` on frozen `{PRETRAINED['mms_base']}`
- Parallel text: `sdmy/kokborok` (SMOL + local dialect correction) and Adivaani `trp`

**Cascade we train**

`Kokborok audio → Kokborok transcript → English`

Cross-lingual start: MMS-1B (massively multilingual speech) + optional Bodo/Bengali Whisper as *areal* prior, not as a genetic parent.

### 2. Bengali (bn)

- {BENGALI.family}
- BanSpeech spontaneous ASR · BhasaAnuvaad / IndicVoices-ST `indic2en` · NPTEL · WordProject
- Direct ST: `{PRETRAINED['indic_seamless']}`

### 3. Marathi (mr)

- {MARATHI.family}
- Same BhasaAnuvaad Indic→English configs (`mr_text` / WordProject / NPTEL)

Fine-tune locally:

```bash
python -m indic_st.train_asr --dataset kokborok_asr --init-from openai/whisper-small
python -m indic_st.train_mt --dataset kokborok_mt --init-from facebook/nllb-200-distilled-600M
```
"""

DATASET_CHOICES = list(DATASETS.keys())

CSS = """
.gradio-container {font-family: 'IBM Plex Sans', system-ui, sans-serif;}
footer {display:none !important;}
"""

with gr.Blocks(title="Tripura ST Lab — Kokborok → English", css=CSS, theme=gr.themes.Soft(primary_hue="orange", neutral_hue="stone")) as demo:
    gr.Markdown(
        "# Tripura ST Lab\n"
        "**Kokborok speech translation** as the first experiment: ASR then MT, "
        "with Bengali and Marathi BhasaAnuvaad tracks beside it.\n\n"
        "Author: Vivek Das · Hugging Face: [vivekharry](https://huggingface.co/vivekharry)"
    )
    with gr.Tab("Kokborok cascade"):
        gr.Markdown("Record 16 kHz mono (or upload WAV). The Space will try NE-ASR + NLLB; CPU Spaces often fall back and tell you to train locally.")
        audio = gr.Audio(type="filepath", label="Kokborok speech")
        btn = gr.Button("Transcribe + translate", variant="primary")
        out_tr = gr.Textbox(label="Kokborok transcript")
        out_en = gr.Textbox(label="English")
        out_meta = gr.Textbox(label="Models")
        btn.click(run_cascade, audio, [out_tr, out_en, out_meta])
    with gr.Tab("Datasets"):
        name = gr.Dropdown(DATASET_CHOICES, value="kokborok_asr", label="Registered corpus")
        n = gr.Slider(1, 12, value=4, step=1, label="Peek rows (streaming)")
        peek = gr.Button("Stream a few rows")
        md = gr.Markdown()
        peek.click(peek_dataset, [name, n], md)
        gr.Markdown(
            "\n".join(
                f"- `{s.hub_id}` · {s.kind} · {s.language} — {s.why}"
                for s in DATASETS.values()
            )
        )
    with gr.Tab("Transfer recipe"):
        gr.Markdown(LANG_MD)
    with gr.Tab("Score a pair"):
        ref = gr.Textbox(label="Reference")
        hyp = gr.Textbox(label="Hypothesis")
        sc = gr.Button("Score")
        out_s = gr.Textbox(label="WER / overlap")
        sc.click(score, [ref, hyp], out_s)

if __name__ == "__main__":
    demo.launch()
