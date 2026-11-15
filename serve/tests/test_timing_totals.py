"""Request timing scopes across engine runs and HTTP reasoning continuations; no GPU."""
import json
import threading
import unittest
from unittest import mock

from serve.engine.timings import RunTotals
from serve.service import request_timings
from serve.tests import test_server as server_tests, test_parallel as parallel_tests


class RunAccounting(unittest.TestCase):
    def test_times_and_counters_sum_but_cache_reuse_is_from_the_first_run(self):
        runs = RunTotals()
        for index in range(3):
            runs.add({"generated": 4, "prompt_ms": 5.0, "decode_ms": 20.0,
                      "reused": 7 if not index else 100, "hits": 2, "lookups": 3,
                      "drafts_accepted": 1, "drafts_offered": 2, "finish": "stop"})
        got = runs.result()
        self.assertEqual((got["prompt_ms"], got["decode_ms"], got["generated"], got["reused"]), (15, 60, 12, 7))
        self.assertEqual((got["hits"], got["lookups"], got["drafts_accepted"], got["drafts_offered"]), (6, 9, 3, 6))
        self.assertEqual(runs.result(), got)                 # reading a summary cannot double-count a run

    def test_missing_counters_do_not_become_partial_totals(self):
        runs = RunTotals()
        runs.add({"generated": 2, "prompt_ms": 5, "decode_ms": 10, "drafts_offered": 4})
        runs.add({"generated": 3, "prompt_ms": 1, "decode_ms": 20})
        self.assertNotIn("drafts_offered", runs.result())
        runs.add({"generated": 1, "prompt_ms": 1, "decode_ms": 5, "drafts_offered": 2})
        self.assertNotIn("drafts_offered", runs.result())
        runs.add({})                                        # no DONE for a pass: its timings are unknown
        self.assertIsNone(request_timings(100, 6, runs.result()))

    def test_zero_engine_output_is_not_replaced_by_server_inserted_tokens(self):
        got = request_timings(100, 5, {"prompt_ms": 10, "decode_ms": 0, "generated": 0})
        self.assertEqual(got["predicted_n"], 0)
        self.assertIsNone(got["predicted_per_second"])


class TimedThinkingEngine(server_tests.ThinkingEngine):
    def __init__(self, tok):
        super().__init__(tok)
        self.runs = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        first = not self.runs
        run = {"generated": 0, "prompt_tokens": len(ids), "prompt_ms": 5000.0 if first else 0.2,
               "decode_ms": 2000.0 if first else 400.0, "finish": "stop", "prompt_read": len(ids),
               "reused": 0 if first else len(self.prompts[-1]), "drafts_accepted": 2, "drafts_offered": 3}
        self.runs.append(run)
        try:
            for token in super().generate(ids, max_new, sampling, cancel, embeddings, session):
                run["generated"] += 1
                yield token
        finally:
            self.last = dict(run)


class ContinuationAccounting(unittest.TestCase):
    def start(self, ended=False):
        case = server_tests.ThinkingBudget()
        case.setUp()
        self.addCleanup(case.tearDown)
        engine = TimedThinkingEngine(case.tok)
        engine.end_in_thought = ended
        case.engine = case.svc.engine = engine
        return case, engine

    def check(self, case, engine, body):
        self.assertEqual(len(engine.runs), 2)
        timing = body["timings"]
        self.assertAlmostEqual(timing["prompt_ms"], 5000.2)
        self.assertAlmostEqual(timing["predicted_ms"], 2400.0)
        self.assertEqual(timing["cache_n"], 0)
        self.assertEqual(timing["prompt_n"], body["usage"]["prompt_tokens"])
        generated = sum(run["generated"] for run in engine.runs)
        self.assertEqual(timing["predicted_n"], generated)
        self.assertEqual(timing["predicted_per_second"], round(generated / 2.4, 1))
        self.assertEqual((timing["draft_n"], timing["draft_n_accepted"]), (6, 4))
        self.assertEqual(case.svc.totals["prompt_ms"], 5000.2)
        self.assertEqual(case.svc.totals["decode_ms"], 2400.0)
        self.assertEqual(case.svc.v1_status()["last_timings"]["prompt_ms"], timing["prompt_ms"])
        self.assertEqual(case.svc.history[-1]["prompt_ms"], timing["prompt_ms"])

    def test_budget_continuation_sums_both_passes(self):
        case, engine = self.start()
        status, body = case.openai(reasoning_budget_tokens=20)
        self.assertEqual(status, 200)
        self.check(case, engine, body)
        self.assertGreater(body["usage"]["completion_tokens"], body["timings"]["predicted_n"])

    def test_streamed_budget_continuation_sums_both_passes(self):
        case, engine = self.start()
        status, raw = case.openai(reasoning_budget_tokens=20, stream=True, stream_options={"include_usage": True})
        self.assertEqual(status, 200)
        chunks = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith("data: {")]
        last = next(chunk for chunk in reversed(chunks) if chunk.get("timings"))
        self.check(case, engine, last)

    def test_ending_a_turn_in_thinking_sums_the_answer_continuation(self):
        case, engine = self.start(ended=True)
        status, body = case.openai(reasoning_budget_tokens=0)
        self.assertEqual(status, 200)
        self.check(case, engine, body)

    def test_stop_string_closes_and_accounts_the_final_pass(self):
        case, engine = self.start()
        status, body = case.openai(reasoning_budget_tokens=20, stop=["answer is"])
        self.assertEqual(status, 200)
        self.check(case, engine, body)

    def test_client_disconnect_accounts_the_closed_pass_once(self):
        case, engine = self.start()
        # Drive Service directly so GeneratorExit occurs during the answer pass, not after HTTP buffering.
        ids = case.tok.encode("<think>", parse_special=True)
        gen = case.svc.run(ids, True, None, 400, {"reasoning_budget_tokens": 20}, threading.Event())
        try:
            for kind, value in gen:
                if len(engine.runs) == 2 and kind == "event":
                    break
        finally:
            gen.close()
        self.assertEqual(len(engine.runs), 2)
        self.assertEqual(case.svc.totals["prompt_ms"], 5000.2)
        self.assertEqual(case.svc.totals["decode_ms"], 2400.0)


class ProtocolAccounting(unittest.TestCase):
    def test_solo_batch_solo_sums_all_protocol_intervals_for_each_request(self):
        case = parallel_tests.ParallelService()
        self.addCleanup(case.tearDown)
        case.start(2, counts=True, slot_cache=True, pms=True)
        events = {}
        parse_done, bdone = case.engine._parse_done, case.engine._bdone_last

        def done(line):
            fields = line.split()
            events.setdefault(threading.get_ident(), []).append(("DONE", int(fields[1]), float(fields[3]), float(fields[4])))
            parse_done(line)

        def batch_done(fields):
            events.setdefault(threading.get_ident(), []).append(("BDONE", int(fields[2]), 0.0, float(fields[4])))
            bdone(fields)

        with mock.patch.object(case.engine, "_parse_done", done), mock.patch.object(case.engine, "_bdone_last", batch_done):
            got = case.timed_chats(["blocker LONGREPLY", "second one"], gap=0.1, max_tokens=400)
        self.assertEqual(len(events), 2)
        expected = []
        for runs in events.values():
            generated, admission = 0, 0
            for kind, count, _, _ in runs:
                if kind == "DONE":
                    generated += count
                    admission = count
                else:
                    generated += count - admission      # BDONE includes its admission's first token
            expected.append((generated, round(sum(run[2] for run in runs), 1), round(sum(run[3] for run in runs), 1)))
        self.assertCountEqual([(t["predicted_n"], t["prompt_ms"], t["predicted_ms"]) for t in got], expected)
        for timing in got:
            self.assertEqual(timing["predicted_per_second"], round(timing["predicted_n"] * 1000 / timing["predicted_ms"], 1))
        self.assertFalse(any(case.engine.slot_busy))
