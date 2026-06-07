import unittest

from fact_check_parsing import VERDICT_LABELS, normalize_claim, parse_verdict


class NormalizeClaimTests(unittest.TestCase):
    def test_plain_text(self):
        self.assertEqual(normalize_claim("Tyrannosaurus was carnivorous."), "Tyrannosaurus was carnivorous.")

    def test_python_string_literal(self):
        self.assertEqual(normalize_claim("'Feathered dinosaurs existed.'"), "Feathered dinosaurs existed.")

    def test_whitespace_trim(self):
        self.assertEqual(normalize_claim("  Spinosaurus  "), "Spinosaurus")

    def test_invalid_literal_falls_back(self):
        self.assertEqual(normalize_claim("'unclosed"), "'unclosed")


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


if __name__ == "__main__":
    unittest.main()
