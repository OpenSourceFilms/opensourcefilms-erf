#!/usr/bin/env python3
"""Provisional person-level Axis-2 telemetry for the ASR bake-off (D-Log-41 decision 2).

REPORTING ONLY — never a baseline, never canon (T-ERF-09: telemetry, not a verdict;
the reviewer's ear stays the authority on truth). Scores person_id/anonymous/not_person_ids
golden assertions by TIME OVERLAP, dropping the speaker_label/stream governance filter,
per arm. Surfaces windows that are gate-empty under governance yet person-correct without
it (the ag_ab5460da species). Flags any window where dropping the filter changes semantics:
multi-assertion partitioned by label/stream, OR overlapping words spanning >1 speaker_label.
Read-only; touches no catalog or canonical surface. Writes a clearly-labelled provisional MD.
"""
import json
from pathlib import Path

ERF = Path(__file__).resolve().parent.parent
ARMS = ["control", "canary", "parakeet", "voxtral"]
GOLDEN = ERF / "state/eval/attribution_golden.jsonl"
OUT = ERF / "state/eval/asr_review/axis2_provisional.md"
PROBE = {"m_5a8bedaa054a", "m_e1b299626700", "m_0e36ebd54774", "m_01d094906b30", "m_9caf6ce31df4"}
THR = 0.90


def governs(w, a):
    if "stream" in a and w.get("stream", "A") != a["stream"]:
        return False
    if "speaker_label" in a and w.get("speaker_label") != a["speaker_label"]:
        return False
    return True


def passes(a, w):
    if "person_id" in a:
        return w.get("person_id") == a["person_id"]
    if "anonymous" in a:
        return w.get("person_id") is None
    return w.get("person_id") not in a["not_person_ids"]


def score(words, A, governed):
    n = c = 0
    for w in words:
        applic = [a for a in A if governs(w, a)] if governed else list(A)
        if not applic:
            continue
        n += 1
        if all(passes(a, w) for a in applic):
            c += 1
    return n, c


golden = [json.loads(l) for l in open(GOLDEN, encoding="utf-8") if l.strip()]
G = sorted([g for g in golden if g.get("verified") and g["media_id"] in PROBE],
           key=lambda x: (x["stratum"], x["media_id"], x["start_ms"]))

L = ["# ASR bake-off — provisional person-level Axis-2 telemetry (D-Log-41 decision 2)",
     "",
     "**PROVISIONAL / REPORTING ONLY — NOT a baseline, NOT canon.** Person-level reading:",
     "person_id/anonymous/not_person_ids assertions scored by time overlap with the",
     "`speaker_label`/`stream` governance filter DROPPED. Purpose: surface windows that are",
     "gate-empty under governance yet person-correct without it (the ag_ab5460da species).",
     "flag = dropping the filter changes semantics (multi-assertion, or overlapping words span",
     ">1 speaker_label) — read those with care; the reviewer's ear (Axis-1 pack) is the authority.",
     ""]

summ = []
for arm in ARMS:
    tx = {}
    for mid in PROBE:
        p = ERF / f"state/bakeoff/asr_shadow/{arm}/catalog/asr/{mid}.transcript.jsonl"
        tx[mid] = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []
    L += [f"## arm: {arm}", "",
          "| golden_id | str | media | gov n/ok | prov n/ok | prov acc | recovers | flag |",
          "|---|---|---|---|---|---|---|---|"]
    gov_empty = recov = 0
    gsc = gco = psc = pco = 0
    for g in G:
        mid, s, e, A = g["media_id"], g["start_ms"], g["end_ms"], g["expected"]
        ww = [w for w in tx[mid] if w["start_ms"] < e and w["end_ms"] > s]
        gn, gc = score(ww, A, True)
        pn, pc = score(ww, A, False)
        labels = {w.get("speaker_label") for w in ww}
        flag = len(A) > 1 or len(labels) > 1
        pacc = f"{pc / pn:.2f}" if pn else "-"
        recovered = gn == 0 and pn > 0 and pc / pn >= THR
        if gn == 0:
            gov_empty += 1
        if recovered and not flag:
            recov += 1
        if not flag:
            gsc += gn; gco += gc; psc += pn; pco += pc
        L.append(f"| {g['golden_id']} | {g['stratum']} | {mid[:8]} | {gn}/{gc} | {pn}/{pc} "
                 f"| {pacc} | {'yes' if recovered else ''} | {'FLAG' if flag else ''} |")
    gwa = f"{gco / gsc:.3f}" if gsc else "-"
    pwa = f"{pco / psc:.3f}" if psc else "-"
    L += ["",
          f"- unflagged word_acc (govern -> prov): **{gwa} -> {pwa}**  "
          f"(scored words {gsc} -> {psc})",
          f"- gate-empty windows: **{gov_empty}**; person-correct recoveries (unflagged): **{recov}**",
          ""]
    summ.append((arm, gwa, pwa, gov_empty, recov))

L += ["## Headline (unflagged windows only)", "",
      "| arm | gov word_acc | prov word_acc | gate-empty | recoveries |",
      "|---|---|---|---|---|"]
for arm, gwa, pwa, ge, rc in summ:
    L.append(f"| {arm} | {gwa} | {pwa} | {ge} | {rc} |")
L.append("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
for arm, gwa, pwa, ge, rc in summ:
    print(f"  {arm:9s} gov_wa={gwa} prov_wa={pwa} gate_empty={ge} recoveries={rc}")
