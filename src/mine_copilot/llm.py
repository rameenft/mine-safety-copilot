"""Provider interface for chat + tool calling. Agents talk to `Provider`, never to an SDK.

Messages use a neutral format so swapping Gemini for Claude is a new class, not a rewrite:
  {"role": "user", "text": ...}
  {"role": "assistant", "text": ..., "tool_calls": [{"name", "args"}], "raw": <provider turn>}
  {"role": "tool", "name": ..., "content": <dict>}
"""

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from dotenv import load_dotenv

RETRY_WAIT = {429: 65, 503: 10}  # seconds: quota resets per minute; overload clears fast


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "args": dict}]
    raw: Any = None  # provider-native turn, replayed verbatim (Gemini 3 needs thought signatures)
    usage: dict = field(default_factory=dict)  # {"input": tokens, "output": tokens incl. thinking}


class Provider(Protocol):
    def chat(self, system: str, messages: list[dict], tools: list[dict] | None) -> Reply: ...


class FakeProvider:
    """Returns scripted replies in order; records what it was sent. For offline tests."""

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, system, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.replies.pop(0)


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    load_dotenv()
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class GeminiProvider:
    def __init__(self, model: str | None = None):
        load_dotenv()
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def _contents(self, messages: list[dict]):
        from google.genai import types

        out = []
        for m in messages:
            if m["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part.from_text(text=m["text"])]))
            elif m["role"] == "assistant":
                out.append(m.get("raw") or types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=m["text"] or " ")]
                    + [types.Part.from_function_call(name=c["name"], args=c["args"])
                       for c in m.get("tool_calls", [])],
                ))
            elif m["role"] == "tool":
                part = types.Part.from_function_response(name=m["name"], response=m["content"])
                # Consecutive tool results go in one user turn, matching the model's call turn.
                if out and out[-1].role == "user" and out[-1].parts[0].function_response:
                    out[-1].parts.append(part)
                else:
                    out.append(types.Content(role="user", parts=[part]))
        return out

    def chat(self, system, messages, tools):
        from google.genai import errors, types

        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                           parameters_json_schema=t["parameters"])
                 for t in tools or []]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
        )
        for attempt in range(4):
            try:
                resp = _gemini_client().models.generate_content(
                    model=self.model, contents=self._contents(messages), config=cfg
                )
                break
            except errors.APIError as e:
                # A per-day quota won't reset in a minute; fail fast instead of sleeping.
                if e.code not in (429, 503) or attempt == 3 or "PerDay" in str(e):
                    raise
                print(f"[llm] {e.code}, retrying in {RETRY_WAIT[e.code]}s")
                time.sleep(RETRY_WAIT[e.code])

        content = resp.candidates[0].content if resp.candidates else None
        parts = (content.parts if content else None) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [{"name": p.function_call.name, "args": dict(p.function_call.args or {})}
                 for p in parts if p.function_call]
        u = resp.usage_metadata
        usage = {"input": (u.prompt_token_count or 0) if u else 0,
                 "output": ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0}
        return Reply(text=text.strip(), tool_calls=calls, raw=content, usage=usage)
