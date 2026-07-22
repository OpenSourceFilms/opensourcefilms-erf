"""Central speaker-embedding adapter.

Swap backend in one place — gallery builder, slot embedder, adjudicator all import here.

API
---
    get_embedder(name="redimnet_b6") -> Embedder
    Embedder.name: str
    Embedder.dim: int
    Embedder.embed(wav: np.ndarray) -> np.ndarray  # 1-D float32 mono 16 kHz → L2-normed 1-D
    Embedder.embed_file(path, start_s=None, end_s=None) -> np.ndarray
    cosine(a, b) -> float

Backends
--------
    "redimnet2_b6" — PalabraAI/redimnet2 b6 lm/vb2+vox2_v0, 192-dim (PRODUCTION,
                     D-Identity-04 2026-07-08). NOTE: dataset must be "vb2+vox2_v0"
                     (the flagship b6 checkpoint) — "vox2" alone is a real but
                     weaker checkpoint and was a bake-off bug that understated v2.
    "redimnet_b6"  — IDRnD/ReDimNet b6 ft_lm/vox2, 192-dim (MIT licence; superseded
                     by redimnet2_b6, D-Identity-04 — kept for reference/rollback)
    "wespeaker"    — pyannote/wespeaker-voxceleb-resnet34-LM, 256-dim
    "ecapa2"       — Jenthe/ECAPA2 (HuggingFace), 192-dim (CC-BY-NC-4.0 licence,
                     fine for this non-commercial project). Weights cached locally at
                     weights/ecapa2/ecapa2.pt (TorchScript, loaded via torch.jit.load).
                     Bake-off only — rejected, see DECISIONS.md 2026-07-08.

Models are lazy-loaded on first .embed() call.  TORCH_HOME and HF_HUB_OFFLINE are set
before any torch import so the cache and offline flag are honoured even when this module
is imported at top-level.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# Must be set before any torch import.
os.environ.setdefault("TORCH_HOME", "/workspace/erf/weights/torchhub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16_000
_DIMS: dict[str, int] = {"redimnet2_b6": 192, "redimnet_b6": 192, "wespeaker": 256, "ecapa2": 192}

ECAPA2_WEIGHTS = Path("/workspace/erf/weights/ecapa2/ecapa2.pt")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (need not be pre-normalised)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------

class Embedder:
    """Lazy-loading speaker embedder.  Model is loaded on the first .embed() call."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._fn = None
        self._model = None  # kept alive to prevent GC / GPU eviction

    @property
    def dim(self) -> int:
        """Output embedding dimension."""
        return _DIMS[self.name]

    def _load(self) -> None:
        if self._fn is not None:
            return
        if self.name == "redimnet2_b6":
            self._fn, self._model = _init_redimnet2_b6()
        elif self.name == "redimnet_b6":
            self._fn, self._model = _init_redimnet_b6()
        elif self.name == "wespeaker":
            self._fn, self._model = _init_wespeaker()
        elif self.name == "ecapa2":
            self._fn, self._model = _init_ecapa2()
        else:
            raise ValueError(f"Unknown backend: {self.name!r}")

    def embed(self, wav: np.ndarray) -> np.ndarray:
        """Embed a 1-D float32 mono 16 kHz waveform.

        Returns a 1-D float32 L2-normalised array of length self.dim.
        """
        self._load()
        return self._fn(wav)

    def embed_file(
        self,
        path: "str | Path",
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
    ) -> np.ndarray:
        """Load audio via torchaudio, resample to 16 kHz mono, crop, and embed.

        Args:
            path: Audio file (wav, mp4, …).
            start_s: Crop start in seconds (None = beginning).
            end_s:   Crop end   in seconds (None = end of file).
        """
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)
        wav = waveform.squeeze(0).numpy().astype(np.float32)

        if start_s is not None or end_s is not None:
            s = int((start_s or 0.0) * SR)
            e = int(end_s * SR) if end_s is not None else len(wav)
            wav = wav[s:e]

        return self.embed(wav)


# ---------------------------------------------------------------------------
# Backend initialisers — return (embed_fn, model_handle)
# ---------------------------------------------------------------------------

def _init_redimnet2_b6():
    """Load PalabraAI/redimnet2 b6 lm/vb2+vox2_v0 from local torch hub cache.

    dataset MUST be "vb2+vox2_v0" (redimnet2's flagship b6 checkpoint, Vox1-O
    EER 0.23%) — plain "vox2" is a real but weaker checkpoint (EER 0.29%) and
    using it was a bake-off bug (2026-07-08) that made v2 look worse than v1
    when in fact the corrected checkpoint wins on aggregate accuracy.
    """
    import torch
    model = torch.hub.load(
        "PalabraAI/redimnet2",
        "redimnet2",
        model_name="b6",
        train_type="lm",
        dataset="vb2+vox2_v0",
        trust_repo=True,
    ).eval().cuda()

    def fn(wav: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).cuda()  # [1, T] — NOT [1,1,T]
        with torch.no_grad():
            emb = model(x)
        return _l2(emb.float().cpu().numpy().reshape(-1))

    return fn, model


def _init_redimnet_b6():
    """Load IDRnD/ReDimNet b6 ft_lm/vox2 from local torch hub cache. Superseded
    by _init_redimnet2_b6 (D-Identity-04) — kept for reference/rollback."""
    import torch
    model = torch.hub.load(
        "IDRnD/ReDimNet",
        "ReDimNet",
        model_name="b6",
        train_type="ft_lm",
        dataset="vox2",
        trust_repo=True,
    ).eval().cuda()

    def fn(wav: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).cuda()  # [1, T] — NOT [1,1,T]
        with torch.no_grad():
            emb = model(x)
        return _l2(emb.float().cpu().numpy().reshape(-1))

    return fn, model


def _init_wespeaker():
    """Load pyannote/wespeaker-voxceleb-resnet34-LM for whole-file inference."""
    import torch
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole", device=torch.device("cuda"))
    _store = {"inference": inference, "model": model}

    def fn(wav: np.ndarray) -> np.ndarray:
        import torch
        # pyannote Inference accepts {"waveform": [1, T], "sample_rate": int}
        waveform = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)  # [1, T]
        result = _store["inference"]({"waveform": waveform, "sample_rate": SR})
        vec = (result if isinstance(result, np.ndarray) else np.asarray(result.data)).flatten()
        return _l2(vec.astype(np.float32))

    return fn, _store


def _init_ecapa2():
    """Load Jenthe/ECAPA2 TorchScript weights from local cache.

    Weights are pre-downloaded to ECAPA2_WEIGHTS (huggingface_hub, repo
    Jenthe/ECAPA2, file ecapa2.pt — CC-BY-NC-4.0, non-commercial). Model is a
    torch.jit.ScriptModule; loaded in float32 (fp16 breaks — cuFFT rejects the
    non-power-of-two STFT window size used internally).
    """
    import torch

    if not ECAPA2_WEIGHTS.exists():
        raise FileNotFoundError(
            f"ECAPA2 weights not found at {ECAPA2_WEIGHTS}. Download via: "
            "huggingface_hub.hf_hub_download(repo_id='Jenthe/ECAPA2', "
            "filename='ecapa2.pt', local_dir='/workspace/erf/weights/ecapa2')"
        )
    model = torch.jit.load(str(ECAPA2_WEIGHTS), map_location="cuda").cuda()

    def fn(wav: np.ndarray) -> np.ndarray:
        import torch
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).cuda()  # [1, T]
        with torch.no_grad():
            emb = model(x)
        return _l2(emb.float().cpu().numpy().reshape(-1))

    return fn, model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(name: str = "redimnet2_b6") -> Embedder:
    """Return a lazy Embedder for the named backend.

    Known backends: "redimnet2_b6" (192-dim, PRODUCTION), "redimnet_b6" (192-dim,
    superseded), "wespeaker" (256-dim), "ecapa2" (192-dim, bake-off only).
    """
    if name not in _DIMS:
        raise ValueError(f"Unknown embedder {name!r}. Known: {sorted(_DIMS)}")
    return Embedder(name)
