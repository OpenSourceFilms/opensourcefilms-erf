#!/usr/bin/env python3
"""Generated ID index + uniqueness guard for DECISIONS.md (D-Meta-01).

An entry HEADER defines the D-IDs appearing before the first ':' on a
'##'/'###'/'####' line. 'addendum'/'RESOLVED'-style headers amend an existing
entry and define nothing. Duplicate definitions are a defect (D-Meta-01):
never reuse an ID — suffix or new namespace, and record renames in a D-Meta entry.

Usage: decisions_index.py [--write]   (default: check mode, exit 1 on
duplicate IDs or a stale index block)
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "DECISIONS.md"
BEGIN, END = "<!-- BEGIN GENERATED ID INDEX (bin/decisions_index.py) -->", "<!-- END GENERATED ID INDEX -->"
ID_RE = re.compile(r"D-[A-Za-z0-9]+-\d+[a-z]?\b")
AMEND_RE = re.compile(r"D-[A-Za-z0-9]+-\d+[a-z]?\s+(addendum|RESOLVED|CORRECTED|amendment|update)", re.I)
DATE_RE = re.compile(r"(20\d\d-\d\d-\d\d)")


def parse(lines):
    defs, dupes, last_date = [], [], ""
    for line in lines:
        if not re.match(r"^#{2,4} ", line):
            continue
        m = DATE_RE.search(line)
        if m:
            last_date = m.group(1)
        head = line.split(":", 1)[0]
        if AMEND_RE.search(head):
            continue
        title = (line.split(":", 1)[1].strip() if ":" in line else line.lstrip("# ").strip())
        title = re.sub(r"\[was [^\]]*\]", "", title).strip()
        for did in ID_RE.findall(head):
            if did in {d for d, _, _ in defs}:
                dupes.append(did)
            defs.append((did, last_date, title[:90]))
    return defs, dupes


def render(defs):
    body = [BEGIN, "", "> Do not edit — regenerate with `bin/decisions_index.py --write`. "
            "New entries MUST use a fresh ID (D-Meta-01).", ""]
    body += [f"- `{d}` — {date or '?'} — {title}" for d, date, title in defs]
    return "\n".join(body + ["", END])


def main():
    text = LEDGER.read_text()
    lines = text.splitlines()
    defs, dupes = parse(lines)
    if dupes:
        print(f"DUPLICATE decision IDs defined more than once: {sorted(set(dupes))}")
        return 1
    block = render(defs)
    if BEGIN in text:
        new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:  # first run: insert after the preamble's first '---'
        new = text.replace("\n---\n", f"\n---\n\n{block}\n", 1)
    if "--write" in sys.argv:
        if new != text:
            LEDGER.write_text(new)
            print(f"index written: {len(defs)} IDs")
        else:
            print(f"index current: {len(defs)} IDs")
        return 0
    if new != text:
        print("index block STALE (or missing) — run: bin/decisions_index.py --write")
        return 1
    print(f"ok: {len(defs)} unique IDs, index current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
