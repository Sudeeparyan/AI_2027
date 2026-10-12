"""Test small lab helpers without importing model libraries or downloading weights.

Run: .venv/Scripts/python.exe tools/test_lab_helpers.py
Only the selected function bodies are extracted from the notebook source files.
"""
from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def extract(week: int, name: str, namespace: dict):
    source = (ROOT / f"curriculum/labs/week_{week:02d}_lab.py").read_text(encoding="utf-8")
    source = "\n".join("pass" if line.startswith("%") else line for line in source.splitlines())
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "curriculum/labs"), "exec"), namespace)
    return namespace[name]


class StreamingTests(unittest.TestCase):
    def measure(self, chunks: list[str], timestamps: list[float]):
        clock = iter(timestamps)
        namespace = {
            "time": SimpleNamespace(perf_counter=lambda: next(clock)),
            "stream_chat": lambda messages, limit: iter(chunks),
            "tok": lambda text, add_special_tokens=False: {"input_ids": text.split()},
        }
        return extract(11, "measure", namespace)([{"role": "user", "content": "test"}], 8)

    def test_no_chunks_has_no_first_token_event(self):
        result = self.measure([], [2.0, 2.3])
        self.assertIsNone(result["TTFT (s)"])
        self.assertEqual(result["tokens/s"], 0.0)
        self.assertEqual(result["answer"], "")

    def test_whitespace_only_has_no_first_token_event(self):
        result = self.measure(["", " ", "\n"], [1.0, 1.5])
        self.assertIsNone(result["TTFT (s)"])
        self.assertEqual(result["tokens/s"], 0.0)
        self.assertEqual(result["output tokens"], 0)
        self.assertEqual(result["total (s)"], 0.5)

    def test_normal_stream_keeps_observed_timing(self):
        result = self.measure(["", "Hello", " world"], [0.0, 0.2, 0.8])
        self.assertEqual(result, {"TTFT (s)": 0.2, "total (s)": 0.8, "output tokens": 2,
                                  "tokens/s": 2.5, "answer": "Hello world"})

    def test_single_token_has_whole_duration_throughput(self):
        result = self.measure(["Hello"], [0.0, 0.1, 0.3])
        self.assertEqual(result["TTFT (s)"], 0.1)
        self.assertEqual(result["tokens/s"], 3.3)

    def test_buffering_changes_first_text_but_not_whole_duration_rate(self):
        early = self.measure(["hello world third"], [0.0, 1.0, 3.0])
        buffered = self.measure(["", "hello world third"], [0.0, 2.0, 3.0])
        self.assertEqual(early["TTFT (s)"], 1.0)
        self.assertEqual(buffered["TTFT (s)"], 2.0)
        self.assertEqual(early["tokens/s"], 1.0)
        self.assertEqual(buffered["tokens/s"], 1.0)


class CacheTests(unittest.TestCase):
    def test_cache_identity_includes_generation_settings_and_model(self):
        calls = []
        namespace = {"CACHE": {}, "MODEL_VERSION": "test-model@fp32", "hashlib": hashlib, "json": json}

        def generate(messages, max_new_tokens):
            request = (namespace["MODEL_VERSION"], messages, max_new_tokens)
            calls.append(request)
            return (json.dumps(request, sort_keys=True), 1, 1)

        namespace["generate"] = generate
        cached = extract(11, "cached_generate", namespace)
        messages = [{"role": "user", "content": "hello"}]
        first = cached(messages, 8)
        self.assertEqual(cached([{"content": "hello", "role": "user"}], 8), first)
        self.assertEqual(len(calls), 1)
        longer = cached(messages, 16)
        self.assertNotEqual(longer, first)
        self.assertEqual(cached(messages, 16), longer)
        self.assertEqual(len(calls), 2)
        namespace["MODEL_VERSION"] = "next-model@fp32"
        next_model = cached(messages, 16)
        self.assertNotEqual(next_model, longer)
        self.assertEqual(len(calls), 3)
        changed_message = cached([{"role": "user", "content": "different"}], 16)
        self.assertNotEqual(changed_message, next_model)
        self.assertEqual(len(calls), 4)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        namespace = {
            # The helper uses only scalar means; keep these tests dependency-free.
            "np": SimpleNamespace(mean=lambda values: sum(values) / len(values)),
            # IDs 0 and 3 share a document, but only 0 contains annotated evidence.
            "CHUNKS": [{"doc": "target"}, {"doc": "other"}, {"doc": "second"}, {"doc": "target"}],
            "EVAL": [("hit", "", "target"), ("miss", "", "second"), ("unanswerable", None, None)],
            "GOLD_PASSAGES": {"hit": {0}, "miss": {2}},
        }
        self.passage_metric = extract(12, "passage_metrics", namespace)
        self.metric = extract(12, "evaluate_retrieval", namespace)

    def test_missing_gold_in_truncated_ranking_contributes_zero(self):
        result = self.metric(lambda query: [1, 0] if query == "hit" else [1], ks=(1, 2))
        self.assertEqual(result, {"recall@1": 0.0, "recall@2": 0.5, "MRR": 0.25})

    def test_empty_rankings_contribute_zero(self):
        result = self.metric(lambda query: [], ks=(1, 3))
        self.assertEqual(result, {"recall@1": 0.0, "recall@3": 0.0, "MRR": 0.0})

    def test_successful_passage_ranks_and_missing_hits(self):
        result = self.metric(lambda query: [0, 2] if query == "hit" else [1, 2, 0], ks=(1, 3))
        self.assertEqual(result, {"recall@1": 0.5, "recall@3": 1.0, "MRR": 0.75})

    def test_same_document_without_evidence_is_not_a_hit(self):
        result = self.metric(lambda query: [3], ks=(1, 3))
        self.assertEqual(result, {"recall@1": 0.0, "recall@3": 0.0, "MRR": 0.0})

    def test_recall_denominator_counts_all_relevant_passages(self):
        result = self.passage_metric([9, 0, 0], {0, 2}, ks=(1, 2, 3))
        self.assertEqual(result, {"recall@1": 0.0, "recall@2": 0.5, "recall@3": 0.5, "MRR": 0.5})

    def test_unanswerable_queries_are_excluded_from_the_mean(self):
        seen = []

        def retrieve(query):
            seen.append(query)
            return [0] if query == "hit" else [2]

        self.assertEqual(self.metric(retrieve, ks=(1,)), {"recall@1": 1.0, "MRR": 1.0})
        self.assertEqual(seen, ["hit", "miss"])

    def test_unlabelled_passage_recall_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one labelled relevant passage"):
            self.passage_metric([0], set())


if __name__ == "__main__":
    unittest.main()
