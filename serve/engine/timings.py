"""Request-local accounting across completed engine runs."""
from __future__ import annotations


class RunTotals:
    """Sum only counters known for every run; retain the first run's external cache reuse.

    The latest run still supplies its finish and prompt extent. Continuations reuse their own
    earlier tokens, which must not become cache hits attributed to an earlier request.
    """

    SUMS = ("prompt_ms", "decode_ms", "generated", "prompt_read", "drafts_accepted", "drafts_offered",
            "hits", "lookups", "file_reads", "offloaded")

    def __init__(self):
        self.last = {}
        self.totals = {}
        self.unknown = set()
        self.reused = None
        self.started = False

    def add(self, last: dict) -> None:
        if not self.started:
            self.reused = last.get("reused")
            self.started = True
        self.last = dict(last)
        for key in self.SUMS:
            value = last.get(key)
            if value is None:
                self.unknown.add(key)
                self.totals.pop(key, None)
            elif key not in self.unknown:
                self.totals[key] = self.totals.get(key, 0) + value

    def result(self) -> dict:
        out = {key: value for key, value in self.last.items() if key not in self.SUMS and key != "reused"}
        out.update(self.totals)
        if self.reused is not None:
            out["reused"] = self.reused
        return out
