from __future__ import annotations

from typing import Any, Iterator

from indic_st.registry import DATASETS, DatasetSpec


def _guess_columns(example: dict[str, Any], spec: DatasetSpec) -> tuple[str | None, str | None]:
    src = spec.source_text if spec.source_text in example else None
    tgt = spec.target_text if spec.target_text and spec.target_text in example else None
    if src is None:
        for key in ("text", "transcription", "sentence", "kokborok", "src", "tribal_text", "bn_text", "mr_text"):
            if key in example:
                src = key
                break
    if tgt is None:
        for key in ("english", "en", "eng_text", "en_text", "translation", "tgt"):
            if key in example:
                tgt = key
                break
    return src, tgt


def load_spec(name: str, streaming: bool = True, split: str | None = None):
    """Load a registered dataset. Streaming by default so laptops survive."""
    from datasets import load_dataset

    spec = DATASETS[name]
    kwargs: dict[str, Any] = {"path": spec.hub_id, "streaming": streaming}
    if spec.config:
        kwargs["name"] = spec.config
    chosen = split or spec.splits[0]
    try:
        return load_dataset(**kwargs, split=chosen), spec
    except Exception as first:
        # Config or split names vary across BhasaAnuvaad dumps.
        try:
            return load_dataset(spec.hub_id, streaming=streaming, split=chosen), spec
        except Exception as second:
            raise RuntimeError(
                f"Could not load {spec.hub_id} ({name}). "
                f"First error: {first}. Retry error: {second}"
            ) from second


def iter_pairs(name: str, limit: int = 8) -> Iterator[dict[str, Any]]:
    ds, spec = load_spec(name, streaming=True)
    n = 0
    for row in ds:
        src_key, tgt_key = _guess_columns(row, spec)
        yield {
            "audio": row.get(spec.audio) if spec.audio else None,
            "source": row.get(src_key) if src_key else None,
            "target": row.get(tgt_key) if tgt_key else None,
            "raw_keys": list(row.keys()),
            "spec": spec.hub_id,
        }
        n += 1
        if n >= limit:
            break
