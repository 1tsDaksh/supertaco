"""Synthetic dataset generation: chunking, dedupe, quality floor (spec G8)."""

import json

from supertaco.data import generate_dataset


class _LLM:
    """Returns scripted generate_rows batches; last entry repeats."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def generate_rows(self, goal, count, banned):
        self.calls.append({"count": count, "banned": list(banned)})
        batch = self.batches[min(len(self.calls) - 1, len(self.batches) - 1)]
        return json.dumps(batch)


def _rows(prefix, n, start=0):
    return [
        {"instruction": f"{prefix} request {i}", "input": "", "output": f"answer {i}"}
        for i in range(start, start + n)
    ]


def test_generate_dataset_chunks_and_stops_at_target():
    llm = _LLM([_rows("a", 50), _rows("b", 50), _rows("c", 50)])
    rows = generate_dataset("goal", llm, target=150, chunk=50)
    assert len(rows) == 150
    assert len(llm.calls) == 3
    assert llm.calls[1]["banned"], "second chunk must see already-used instructions"


def test_generate_dataset_dedupes_across_chunks():
    llm = _LLM([_rows("x", 40), _rows("x", 40) + _rows("y", 20)])
    rows = generate_dataset("g", llm, target=60, chunk=50, floor=50)
    assert len(rows) == 60
    instructions = [r["instruction"] for r in rows]
    assert len(instructions) == len(set(instructions))


def test_generate_dataset_returns_empty_below_floor():
    llm = _LLM([[]])
    rows = generate_dataset("g", llm, target=150, floor=100, max_calls=6)
    assert rows == []
    assert len(llm.calls) == 6  # attempts are bounded, never infinite


def test_generate_dataset_filters_junk_rows():
    llm = _LLM(
        [
            [
                {"instruction": "", "output": "x"},
                {"instruction": "ok one", "output": "   "},
                {"instruction": "good one", "output": "good answer"},
            ]
        ]
    )
    rows = generate_dataset("g", llm, target=1, chunk=10, floor=1, max_calls=1)
    assert [r["instruction"] for r in rows] == ["good one"]


def test_generate_dataset_garbage_reply_returns_empty():
    class Garbage:
        def generate_rows(self, goal, count, banned):
            return "I cannot help with that."

    rows = generate_dataset("g", Garbage(), target=150, floor=100, max_calls=3)
    assert rows == []


def test_generate_dataset_never_raises_when_llm_raises():
    class Boom:
        def generate_rows(self, goal, count, banned):
            raise RuntimeError("token factory down")

    rows = generate_dataset("g", Boom(), target=150, floor=100, max_calls=3)
    assert rows == []
