> Answered by `docs/ERF_Stack_Landscape_Check.md`.

# Deep Research Brief — ERF stack landscape check (July 2026)

**Purpose:** one-shot comparative research report to ground a full re-evaluation of the
ERF media-intelligence stack. Nothing is sacred; we will swap any component — or whole
component *categories* — where something verifiably better exists. Compute, storage,
and bandwidth are free (RunPod GPU pods); build-attention is the scarce resource.

## The system being evaluated (context for every question)

A self-hosted pipeline that ingests behind-the-scenes / field footage of a film
production (~310 videos, growing in regular batches; phone + camera audio, frequently
noisy, off-mic speakers, crosstalk) and produces:

1. **Word-level speaker-attributed transcripts** — who said every word, including
   during overlapping speech; recurring cast (~12 enrolled identities with voice +
   face references) plus unknown one-off people who must get stable anonymous IDs.
2. **Burned-in per-speaker colour-coded captions** on rendered clips.
3. **Semantic retrieval** over the corpus (text + visual embeddings) for human editors.
4. **Automatically assembled narrative short videos** (LLM-planned, evidence-grounded,
   validator-gated) rendered with ffmpeg.

Current components (all open/self-hosted unless noted):
- **ASR:** faster-whisper large-v3 + WhisperX forced alignment + Silero VAD; a targeted
  "recall pass" re-runs plain Whisper on detected drop-out windows.
- **Diarization:** pyannote community-1 (baseline + exclusive modes).
- **Speaker verification / voiceprints:** WeSpeaker voxceleb-resnet34-LM (256-d),
  per-media centroids, cosine gates for adjudicating contested words.
- **Face:** InsightFace buffalo_l; **active speaker detection:** Light-ASD.
- **Embeddings/retrieval:** Qwen3-Embedding-8B (text), Qwen3-VL-Embedding-8B (visual),
  LanceDB + SQLite FTS.
- **LLM layer:** DeepSeek Pro/Flash APIs (grounded editor + cheap labelling).
- **Orchestration:** bespoke Python stage scripts + SQLite journal + explicit outcome
  records; single-operator, batch-only, on MooseFS (no safe concurrent writes to one file).
- **Render:** bespoke ffmpeg/ASS subtitle pipeline.

Hard constraints: English only. Batch (no realtime). Single operator. Word-level
timestamps are non-negotiable (captions + edit points derive from them). Unreleased
film footage — flag any hosted/API option's data-retention posture explicitly.

## Questions (source-backed; prefer 2025–2026 releases/benchmarks; include licence, VRAM, maintenance status, and links)

### A. Speech recognition
1. As of mid-2026, what beats faster-whisper large-v3 for **English word-level-timestamped
   ASR on noisy field audio**? Evaluate at minimum: Whisper large-v3-turbo, NVIDIA
   Parakeet-TDT (v2/v3), NVIDIA Canary/Canary-Qwen, Mistral Voxtral, Kyutai STT, and
   anything newer. WER on noisy/far-field benchmarks (not just LibriSpeech), timestamp
   quality, hallucination behaviour on silence/music.
2. Is WhisperX still the best forced-alignment layer, or have better aligners emerged
   (e.g. improved wav2vec2 variants, NeMo Forced Aligner, torchaudio CTC alignment APIs)?
3. Do any current ASR systems natively output **speaker-attributed words in overlapping
   speech** (target-speaker ASR, e.g. NVIDIA multi-speaker models, SOT-style decoders,
   Meta/other TS-ASR)? Maturity, English performance, word timestamps?

### B. Diarization + overlap
4. Best open diarization for **overlap-heavy real-world video audio** as of mid-2026:
   pyannote community-1 vs anything newer (pyannote 4.x developments, NVIDIA Sortformer
   v2 — note its 4-speaker limit, EEND variants, "VibeVoice" diarization claims — verify
   that one, it appeared in a low-quality source). DER *with* overlap scored
   (collar=0, skip_overlap=False).
5. The paid pyannote **precision-2 API**: measured advantage over community-1, price,
   and data-retention terms.

### C. Speaker verification
6. Best open speaker-embedding model for **verification/identification on short
   (0.5–3 s) noisy segments**: WeSpeaker ResNet34-LM (current, Vox1-O EER 0.72%) vs
   deeper WeSpeaker ResNets (152/221/293), ReDimNet, ECAPA2, W2V-BERT-based models,
   anything newer. Include short-utterance and noisy-condition results, not just clean
   full-utterance VoxCeleb EER.

### D. Faces + active speaker
7. Is InsightFace buffalo_l still the right face detect/recognise stack, or have
   better open models shipped (2025–2026)?
8. Light-ASD vs TalkNet vs newer active-speaker-detection models — accuracy on
   in-the-wild video, maintenance status.

### E. Hosted alternatives for the whole speech layer (honest comparison)
9. Would a hosted STT+diarization API (AssemblyAI Universal-2/latest, Deepgram Nova-3,
   Gladia, Speechmatics, ElevenLabs Scribe, Rev AI) **materially beat** the self-hosted
   stack above on noisy crosstalk footage? Word-level timestamps + diarization quality,
   per-hour cost at ~50–100 h/batch, and **explicit data-retention/training-use terms**
   for unreleased film footage.

### F. System shape (tool-type sets, not just models)
10. For a single-operator, batch, ~10³-asset media pipeline with a file-canonical
    JSON catalog: do asset-oriented orchestrators (Dagster software-defined assets,
    Prefect 3, Temporal, Snakemake) offer enough over bespoke scripts + SQLite journal
    to justify migration? What do comparable open media-AI pipelines actually use?
11. Retrieval: LanceDB + SQLite FTS + Qwen3 8B embedders + reranker — is there a
    verifiably better small-corpus retrieval recipe as of mid-2026 (new embedding
    models topping MTEB/vision benchmarks, hybrid-search defaults)?
12. Auto-editing/render: any mature open tooling for LLM-planned video assembly with
    burned captions that would replace a bespoke ffmpeg/ASS pipeline (e.g. Remotion-class
    programmatic renderers with Python bindings, OTIO-centric toolchains)? Or is bespoke
    ffmpeg still the right call?
13. Anything category-level we're missing: a tool type that systems with the same goal
    (footage → attributed transcript → searchable → auto-edited shorts) commonly use
    that ERF doesn't have at all.

## Output format
Per question: recommendation, runner-up, evidence (benchmark numbers + links + dates),
licence, VRAM/cost, and a one-line "switch worth it? (yes/no/only-if)" verdict against
the incumbent named above.
