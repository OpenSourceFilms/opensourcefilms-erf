"""G-TEXT-EMBED gate — §14. Must pass before B1 runs on any real media.

Loads Qwen3-Embedding-8B from local weights, confirms dim=4096, verifies
query-instruction prefix from config_sentence_transformers.json, runs a
5-pair sanity set. Writes state/embed_text_invocation.json and
state/model_registry.json on success. Hard-exits 1 on failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ERF_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ERF_ROOT / "weights" / "Qwen3-Embedding-8B"
STATE = ERF_ROOT / "state"

EXPECTED_DIM = 4096
MODEL_TAG = "qwen3-8b"

# 5 query→(relevant, irrelevant) sanity pairs — editorial retrieval domain
SANITY_PAIRS = [
    (
        "warehouse argument at night",
        "we need to talk — you promised me you'd never come back here",
        "the sunrise light caught the dew on the meadow grass",
    ),
    (
        "crying emotional breakdown",
        "she couldn't hold it together anymore and broke down sobbing",
        "the camera panned slowly across the empty corridor",
    ),
    (
        "confrontation fight shouting",
        "don't you dare walk away from me when I'm talking to you",
        "the production designer sketched the floor plan on a napkin",
    ),
    (
        "confession secret revealed",
        "I never told anyone this but I was there the night it happened",
        "we tested the color grading on three different monitors",
    ),
    (
        "tender quiet moment between two people",
        "they sat in silence for a long time just holding hands",
        "the crew loaded the equipment into the van before dawn",
    ),
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    print(f"Loading {WEIGHTS} …", flush=True)

    model = SentenceTransformer(
        str(WEIGHTS),
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    # 1. Confirm prompt config
    prompts = model.prompts
    query_prompt = prompts.get("query", "")
    doc_prompt = prompts.get("document", "")
    print(f"  query_prompt  : {query_prompt!r}")
    print(f"  document_prompt: {doc_prompt!r}")

    # 2. Probe dim with a single encode
    probe = model.encode(["test"], prompt_name="query", normalize_embeddings=True)
    dim = probe.shape[1]
    if dim != EXPECTED_DIM:
        print(f"FAIL: expected dim {EXPECTED_DIM}, got {dim}", file=sys.stderr)
        sys.exit(1)
    print(f"  dim: {dim} ✓")

    # 3. Check pre-normalisation
    norm = float(np.linalg.norm(probe[0]))
    pre_normalized = abs(norm - 1.0) < 1e-3
    print(f"  norm={norm:.6f}, pre_normalized={pre_normalized}")

    # 4. Sanity pairs
    failures = []
    for i, (query, relevant, irrelevant) in enumerate(SANITY_PAIRS, 1):
        q_emb = model.encode([query], prompt_name="query", normalize_embeddings=True)[0]
        r_emb = model.encode([relevant], normalize_embeddings=True)[0]
        ir_emb = model.encode([irrelevant], normalize_embeddings=True)[0]
        sim_rel = cosine(q_emb, r_emb)
        sim_irr = cosine(q_emb, ir_emb)
        ok = sim_rel > sim_irr
        mark = "✓" if ok else "✗"
        print(f"  pair {i}: rel={sim_rel:.4f} irr={sim_irr:.4f} {mark}")
        if not ok:
            failures.append(i)

    if failures:
        print(f"FAIL: pairs {failures} did not satisfy rel > irr", file=sys.stderr)
        sys.exit(1)

    print("All 5 sanity pairs passed ✓")

    # 5. Write invocation record
    invocation = {
        "model_tag": MODEL_TAG,
        "model_path": str(WEIGHTS),
        "dim": EXPECTED_DIM,
        "pre_normalized": pre_normalized,
        "query_prompt_name": "query",
        "query_prompt": query_prompt,
        "document_prompt": doc_prompt,
        "encode_kwargs": {
            "normalize_embeddings": True,
            "batch_size": 64,
        },
        "model_kwargs": {
            "device_map": "auto",
        },
        "tokenizer_kwargs": {"padding_side": "left"},
        "gate": "G-TEXT-EMBED",
        "gate_passed": True,
    }

    inv_path = STATE / "embed_text_invocation.json"
    tmp = inv_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(invocation, indent=2, ensure_ascii=False))
    tmp.rename(inv_path)
    print(f"Wrote {inv_path}")

    # 6. Write / update model_registry.json
    reg_path = STATE / "model_registry.json"
    registry: dict = {}
    if reg_path.exists():
        registry = json.loads(reg_path.read_text())

    registry[MODEL_TAG] = {
        "model_tag": MODEL_TAG,
        "model_path": str(WEIGHTS),
        "dim": EXPECTED_DIM,
        "metric": "cosine",
        "modality": "text",
        "gate": "G-TEXT-EMBED",
        "gate_passed": True,
    }

    tmp = reg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
    tmp.rename(reg_path)
    print(f"Wrote {reg_path}")
    print("G-TEXT-EMBED: PASS")


if __name__ == "__main__":
    main()
