"""Editorial-function labelling — layer 1 of the storytelling methodology.

Labels each sentence atom with its narrative role, the production tensions it
embodies, what it can do in a story chain, and whether it's a complete thought.
This precomputes the editorial affordances so the editor/critics reason in story
mechanics (want→obstacle→turn→payoff) instead of thematic similarity.

Flash-first (cheap, bounded). One call per clip. Output consumed by the candidate
generator and the quote-completion repair.
"""
from __future__ import annotations

import json
import logging
import re

from .llm import complete, _FLASH_MODEL

log = logging.getLogger("editorial")

# Narrative-role taxonomy (generic + film-production specific).
FUNCTIONS = [
    "HOOK", "CONTEXT", "DESIRE", "OBSTACLE", "STAKES", "FAILED_ATTEMPT", "DISCOVERY",
    "DECISION", "ESCALATION", "REVERSAL", "JOKE", "REVEAL", "CRAFT_INSIGHT", "PAYOFF",
    "BUTTON", "B_ROLL_SUPPORT",
    "SCRAPPY_SOLUTION", "INDIE_CONSTRAINT", "VFX_PROBLEM", "PRACTICAL_TEST",
    "PERFORMANCE_DISCOVERY", "GEAR_HACK", "LOCATION_PROBLEM", "TEAM_DYNAMIC",
    "DANGEROUS_IDEA_BUT_SAFE", "OPEN_SOURCE_METHOD",
]
TENSIONS = [
    "ambition_vs_budget", "safety_vs_spectacle", "professional_standard_vs_homemade",
    "beautiful_idea_vs_ugly_reality", "simple_plan_vs_unexpected_constraint",
    "ai_vfx_promise_vs_practical_mess", "serious_film_vs_absurd_method", "control_vs_chaos",
]
ARCHETYPES = [
    "THIS_SHOULD_NOT_WORK", "THE_INVISIBLE_PROBLEM", "THE_SAFETY_PARADOX",
    "THE_PROFESSIONAL_VERSION_COSTS_50K", "THE_SHOT_GOT_BIGGER",
    "THE_DUMB_LOOKING_THING_IS_THE_SERIOUS_THING",
]

_SYSTEM = """\
You are a story analyst for a scrappy indie sci-fi film's BTS archive. You do NOT write.
You label documentary sentence atoms with their editorial role so an editor can later
assemble true stories (want → obstacle → decision → turn → payoff), not topic montages.

Use ONLY these FUNCTION labels: {functions}
Use ONLY these TENSION tags: {tensions}
Use ONLY these ARCHETYPE candidates: {archetypes}

Be strict and specific. Most sentences are CONTEXT or B_ROLL_SUPPORT; reserve OBSTACLE,
DECISION, PAYOFF, REVEAL etc. for sentences that genuinely carry that force. quote_completeness
must reflect whether the sentence is a self-contained thought or needs its neighbours to make sense.

Return ONLY strict JSON:
{{
 "clip_level": {{"story_forces": {{"want": str|null, "obstacle": str|null, "decision": str|null,
   "turn": str|null, "payoff": str|null}}, "central_tension": str|null, "archetype_candidates": [str]}},
 "sentences": [{{"sentence_index": int, "functions": [str], "tension_tags": [str],
   "chain_affordances": {{"can_start_story": bool, "can_escalate_story": bool,
     "can_payoff_story": bool, "needs_followup": bool}},
   "quote_completeness": {{"complete_thought": bool, "requires_previous": bool, "requires_next": bool}}}}]
}}\
"""


def _parse_json(raw: str) -> dict:
    if isinstance(raw, dict):
        return raw
    t = (raw or "").strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise


def _format(pkg: dict) -> str:
    head = (f"CLIP {pkg.get('media_id')} | footage={pkg.get('footage_type')} | "
            f"setting={pkg.get('setting')}\nsynopsis: {pkg.get('synopsis')}\n\nSENTENCES:")
    lines = [head]
    for s in pkg.get("sentences", []):
        lines.append(f"  {s['sentence_index']} [{s.get('speaker')}] \"{s['text']}\""
                     f"  (emo={s.get('emotional_beat')} hook={s.get('hook')} quote={s.get('quote')})")
    return "\n".join(lines)


def label_clip(pkg: dict, llm=None) -> dict:
    """Return the editorial-label document for one evidence package."""
    llm = llm or complete
    n = len(pkg.get("sentences", []))
    if n == 0:
        return {"media_id": pkg.get("media_id"), "clip_level": {}, "sentences": []}
    system = _SYSTEM.format(functions=", ".join(FUNCTIONS), tensions=", ".join(TENSIONS),
                            archetypes=", ".join(ARCHETYPES))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": _format(pkg) + "\n\nLabel every sentence. Return ONLY JSON."}]
    raw = llm(messages, json_only=True, model=_FLASH_MODEL)
    doc = _parse_json(raw)
    valid_idx = {s["sentence_index"] for s in pkg["sentences"]}
    sents = [s for s in doc.get("sentences", [])
             if isinstance(s.get("sentence_index"), int) and s["sentence_index"] in valid_idx]
    return {"media_id": pkg.get("media_id"),
            "clip_level": doc.get("clip_level", {}),
            "sentences": sents}
