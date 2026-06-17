#!/usr/bin/env python3
"""Bare terminal chat: prompt line + connected model in the bottom-right corner."""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

DEFAULT_API_BASE = "http://127.0.0.1:3634/v1"
DEFAULT_API_KEY = "local"

SYSTEM_PROMPT = (
    "You are a coding assistant inside the PaleoFactCheck repository. "
    "Keep answers concise. You can create or edit project files when asked."
)


def api_base() -> str:
    url = os.environ.get("LOCAL_MODEL_URL") or os.environ.get("OPENAI_API_BASE") or DEFAULT_API_BASE
    return url.rstrip("/")


def api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("LOCAL_MODEL_API_KEY") or DEFAULT_API_KEY


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"}
    data = json.dumps(payload).encode() if payload is not None else None
    method = "POST" if payload is not None else "GET"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch_model_info() -> tuple[str, str, bool]:
    try:
        body = http_json(f"{api_base()}/models")
        models = body.get("data") or []
        if models:
            model_id = str(models[0].get("id", "default"))
            return model_id, model_id, True
        model_id = str(body.get("model", "default"))
        return model_id, model_id, True
    except Exception:
        fallback = os.environ.get("LOCAL_MODEL_NAME", "not connected")
        return fallback, os.environ.get("LOCAL_MODEL_NAME", "default"), False


def chat(messages: list[dict[str, str]], model: str) -> str:
    body = http_json(
        f"{api_base()}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": int(os.environ.get("CHAT_MAX_TOKENS", "2048")),
            "temperature": float(os.environ.get("CHAT_TEMPERATURE", "0.7")),
        },
        timeout=300,
    )
    return body["choices"][0]["message"]["content"]


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main() -> int:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.styles import Style

    os.chdir(Path(__file__).resolve().parent)

    state = {"label": "connecting...", "model_id": "default", "online": False}

    def refresh_model() -> None:
        state["label"], state["model_id"], state["online"] = fetch_model_info()

    def bottom_toolbar():
        width = shutil.get_terminal_size(fallback=(80, 24)).columns
        label = state["label"]
        style = "class:model-online" if state["online"] else "class:model-offline"
        pad = max(0, width - len(label))
        return FormattedText([(style, (" " * pad) + label)])

    refresh_model()
    clear_screen()

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    session = PromptSession(
        message="> ",
        bottom_toolbar=bottom_toolbar,
        style=Style.from_dict(
            {
                "model-online": "noreverse fg:#6a9955",
                "model-offline": "noreverse fg:#888888",
            }
        ),
    )

    while True:
        try:
            with patch_stdout():
                user_text = session.prompt()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        cmd = user_text.strip().lower()
        if cmd in {"/exit", "/quit"}:
            break
        if cmd == "/clear":
            clear_screen()
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            continue
        if cmd == "/reconnect":
            refresh_model()
            continue
        if not user_text.strip():
            continue

        messages.append({"role": "user", "content": user_text})
        print()

        if not state["online"]:
            refresh_model()
        if not state["online"]:
            messages.pop()
            continue

        try:
            reply = chat(messages, state["model_id"])
        except Exception as exc:
            print(f"Error: {exc}\n")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(reply)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
