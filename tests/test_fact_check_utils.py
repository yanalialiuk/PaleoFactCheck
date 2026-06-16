import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import patch

from fact_check import configure_llm_loader, reset_llm_cache, run_fact_check
from fact_check_parsing import VERDICT_LABELS, normalize_claim, parse_verdict


def _install_fake_retrieval(retrieval_obj):
    """Install a stub data_processing.retrieval into sys.modules so fact_check's
    lazy import picks it up without needing torch/sentence-transformers installed."""
    module = types.ModuleType("data_processing.retrieval")
    module.retrieve_claim_context = lambda *a, **kw: retrieval_obj
    sys.modules["data_processing.retrieval"] = module
    return module


def _uninstall_fake_retrieval():
    sys.modules.pop("data_processing.retrieval", None)


@dataclass
class _FakeChunk:
    text: str
    source: str
    distance: Optional[float] = None


@dataclass
class _FakeRetrievalResult:
    query: str
    chunks: list = field(default_factory=list)

    @property
    def context(self) -> str:
        if not self.chunks:
            return "No relevant information available."
        return "\n".join(chunk.text for chunk in self.chunks)

    @property
    def sources(self) -> list:
        seen: set = set()
        ordered: list = []
        for chunk in self.chunks:
            if chunk.source not in seen:
                seen.add(chunk.source)
                ordered.append(chunk.source)
        return ordered


class NormalizeClaimTests(unittest.TestCase):
    def test_plain_text(self):
        self.assertEqual(normalize_claim("Tyrannosaurus was carnivorous."), "Tyrannosaurus was carnivorous.")

    def test_double_quoted_literal(self):
        self.assertEqual(normalize_claim("'Feathered dinosaurs existed.'"), "Feathered dinosaurs existed.")

    def test_single_quoted_literal(self):
        self.assertEqual(normalize_claim("'Spinosaurus was aquatic.'"), "Spinosaurus was aquatic.")

    def test_whitespace_trim(self):
        self.assertEqual(normalize_claim("  Spinosaurus  "), "Spinosaurus")

    def test_invalid_double_quote_falls_back(self):
        self.assertEqual(normalize_claim('"unclosed'), '"unclosed')

    def test_invalid_single_quote_falls_back(self):
        self.assertEqual(normalize_claim("'unclosed"), "'unclosed")

    def test_non_string_input(self):
        self.assertEqual(normalize_claim(42), "42")


class ParseVerdictTests(unittest.TestCase):
    def test_true_label(self):
        self.assertEqual(parse_verdict("True. The sources support this."), "True")

    def test_false_label(self):
        self.assertEqual(parse_verdict("Answer: False"), "False")

    def test_insufficient_label(self):
        self.assertEqual(parse_verdict("Insufficient information for this claim."), "Insufficient information")

    def test_empty_defaults_to_insufficient(self):
        self.assertEqual(parse_verdict(""), "Insufficient information")

    def test_all_canonical_labels(self):
        for label in VERDICT_LABELS:
            self.assertEqual(parse_verdict(label), label)


class _FakeLLM:
    """Minimal Llama-like callable returning a fixed completion."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.call_count = 0

    def __call__(self, prompt, **kwargs):
        self.call_count += 1
        return {"choices": [{"text": self._text}]}


class RunFactCheckTests(unittest.TestCase):
    def setUp(self):
        reset_llm_cache()

    def tearDown(self):
        configure_llm_loader(None)
        reset_llm_cache()

    def test_empty_claim_short_circuits(self):
        result = run_fact_check("   ")
        self.assertEqual(result.claim, "")
        self.assertEqual(result.verdict, "Insufficient information")
        self.assertEqual(result.sources, [])

    def test_happy_path_returns_fact_check_result(self):
        fake_llm = _FakeLLM("True. The fossil record supports this.")
        configure_llm_loader(lambda: fake_llm)

        retrieval = _FakeRetrievalResult(
            query="T. rex was a carnivore.",
            chunks=[_FakeChunk(text="Tyrannosaurus was a theropod predator.", source="wiki:Tyrannosaurus")],
        )
        _install_fake_retrieval(retrieval)
        try:
            result = run_fact_check("T. rex was a carnivore.", top_k=2)
        finally:
            _uninstall_fake_retrieval()

        self.assertEqual(result.claim, "T. rex was a carnivore.")
        self.assertEqual(result.verdict, "True")
        self.assertEqual(result.sources, ["wiki:Tyrannosaurus"])
        self.assertIn("theropod", result.context_excerpt)
        self.assertEqual(fake_llm.call_count, 1)

    def test_no_chunks_still_returns_verdict(self):
        fake_llm = _FakeLLM("Insufficient information")
        configure_llm_loader(lambda: fake_llm)
        retrieval = _FakeRetrievalResult(query="Mystery claim", chunks=[])
        _install_fake_retrieval(retrieval)
        try:
            result = run_fact_check("Mystery claim")
        finally:
            _uninstall_fake_retrieval()

        self.assertEqual(result.verdict, "Insufficient information")
        self.assertEqual(result.sources, [])


if __name__ == "__main__":
    unittest.main()
