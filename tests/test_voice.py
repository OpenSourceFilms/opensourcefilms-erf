import pytest
from erf.voice import filter_turns


def test_empty_input():
    kept, skipped = filter_turns([])
    assert kept == []
    assert skipped == []


def test_single_turn_kept():
    turn = {"speaker_label": "A", "start_ms": 0, "end_ms": 2000}
    kept, skipped = filter_turns([turn])
    assert len(kept) == 1
    assert skipped == []


def test_single_turn_too_short():
    turn = {"speaker_label": "A", "start_ms": 0, "end_ms": 1999}
    kept, skipped = filter_turns([turn])
    assert kept == []
    assert skipped[0]["skip_reason"] == "turn_too_short"


def test_single_turn_overlap():
    turn = {"speaker_label": "A", "start_ms": 0, "end_ms": 5000, "overlap": True}
    kept, skipped = filter_turns([turn])
    assert kept == []
    assert skipped[0]["skip_reason"] == "overlap"


def test_merge_same_speaker_small_gap():
    turns = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000},
        {"speaker_label": "A", "start_ms": 1800, "end_ms": 3000},
    ]
    kept, skipped = filter_turns(turns, gap_ms=1000, min_ms=2000)
    assert len(kept) == 1
    assert kept[0]["start_ms"] == 0
    assert kept[0]["end_ms"] == 3000
    assert kept[0]["overlap"] is False
    assert skipped == []


def test_no_merge_large_gap():
    turns = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000},
        {"speaker_label": "A", "start_ms": 2500, "end_ms": 3000},
    ]
    kept, skipped = filter_turns(turns, gap_ms=1000, min_ms=2000)
    assert kept == []
    assert len(skipped) == 2
    assert all(t["skip_reason"] == "turn_too_short" for t in skipped)
    assert {t["start_ms"] for t in skipped} == {0, 2500}


def test_different_speaker_no_merge():
    turns = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000},
        {"speaker_label": "B", "start_ms": 1500, "end_ms": 2000},
    ]
    kept, skipped = filter_turns(turns, gap_ms=1000, min_ms=2000)
    assert kept == []
    assert len(skipped) == 2
    assert all(t["skip_reason"] == "turn_too_short" for t in skipped)


def test_overlap_skipped_cannot_be_rescued_by_merge():
    turns = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 1000},
        {"speaker_label": "A", "start_ms": 1500, "end_ms": 2500, "overlap": True},
    ]
    kept, skipped = filter_turns(turns, gap_ms=1000, min_ms=2000)
    assert kept == []
    assert len(skipped) == 2
    assert {t["skip_reason"] for t in skipped} == {"turn_too_short", "overlap"}


def test_clean_turn_kept_alongside_skipped_overlap():
    turns = [
        {"speaker_label": "A", "start_ms": 0, "end_ms": 3000},
        {"speaker_label": "A", "start_ms": 3500, "end_ms": 4000, "overlap": True},
    ]
    kept, skipped = filter_turns(turns, gap_ms=1000, min_ms=2000)
    assert len(kept) == 1
    assert kept[0]["start_ms"] == 0
    assert kept[0]["end_ms"] == 3000
    assert len(skipped) == 1
    assert skipped[0]["skip_reason"] == "overlap"
