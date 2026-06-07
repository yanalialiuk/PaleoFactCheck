import ast
import re

VERDICT_LABELS = ("True", "False", "Insufficient information")


def normalize_claim(raw_query: str) -> str:
    """Accept plain text or a Python string literal from legacy callers."""
    if not isinstance(raw_query, str):
        return str(raw_query).strip()

    stripped = raw_query.strip()
    if stripped.startswith(("'", '"')):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, str):
                return parsed.strip()
        except (ValueError, SyntaxError):
            pass
    return stripped


def parse_verdict(raw_answer: str) -> str:
    cleaned = (raw_answer or "").strip()
    if not cleaned:
        return "Insufficient information"

    for label in VERDICT_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", cleaned, flags=re.IGNORECASE):
            return label

    return cleaned
