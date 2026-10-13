"""Minimal tool-calling clients for the OpenAI, Anthropic and Gemini HTTP APIs.

Uses only the standard library. Each client keeps the full native conversation, including
reasoning items, so every request resends the history.
"""

import base64
import copy
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

# Provider name -> (API key environment variable, default base URL).
PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta"),
}
TIMEOUT = 180  # seconds per HTTP request


class Model:
    """One conversation with a provider model that answers with tool calls.

    Args:
        api: Provider: "openai", "anthropic" or "gemini".
        model: Provider model name.
        api_key: Key to use instead of the provider's environment variable.
        base_url: Endpoint to use instead of the provider's default.
        reasoning: Reasoning effort or level; None leaves it unset.
        max_tokens: Maximum output tokens per response.

    Raises:
        ValueError: If `api` is unknown or no key is available.
    """

    def __init__(self, api, model, *, api_key=None, base_url=None, reasoning=None, max_tokens=4096):
        if api not in PROVIDERS:
            raise ValueError("api must be openai, anthropic, or gemini")
        variable, url = PROVIDERS[api]
        self.key = api_key or os.environ.get(variable)
        if not self.key:
            raise ValueError(f"Set {variable} or pass api_key")
        self.api, self.model = api, model
        self.url = (base_url or url).rstrip("/")
        self.reasoning, self.max_tokens = reasoning, max_tokens
        self.history = []  # native turns sent and received so far
        self.pending = []  # tool calls from the last reply, answered by the next request
        self.usage = {}  # token usage reported for the last reply

    def _post(self, path, payload):
        """POST `payload` as JSON and return the decoded reply; errors never include the key."""
        headers = {"Content-Type": "application/json"}
        if self.api == "openai":
            headers["Authorization"] = f"Bearer {self.key}"
        elif self.api == "anthropic":
            headers.update({"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        else:
            headers["x-goog-api-key"] = self.key
        request = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace").replace(self.key, "[key]")
            raise RuntimeError(
                f"{self.api} HTTP {error.code}; check model access, credentials and quota"
                f"\n{body[:2000]}"
            ) from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError(f"{self.api} request failed or timed out; not retried") from None

    def ask(self, observation, feedback, system, tools):
        """Send one observation and return the tool calls in the reply.

        Args:
            observation: Dict with RGB arrays under "images"; everything else is sent as JSON.
            feedback: Result of the previous tool call, returned to the model as its output.
            system: System prompt.
            tools: Tool definitions with "name", "description" and JSON-schema "parameters".

        Returns:
            A list of {"id", "name", "args"} dicts; "args" is None if it was not valid JSON.
        """
        images = []
        for name, rgb in observation["images"].items():
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            images.append((name, base64.b64encode(buffer.getvalue()).decode()))
        text = json.dumps({key: value for key, value in observation.items() if key != "images"})
        handlers = {"openai": self._openai, "anthropic": self._anthropic, "gemini": self._gemini}
        turns, calls, usage = handlers[self.api](text, images, feedback, system, tools)
        self.history.extend(copy.deepcopy(turns))
        self.pending, self.usage = calls, usage
        return self.pending

    def _openai(self, text, images, feedback, system, tools):
        """One Responses API request; returns (new turns, tool calls, usage)."""
        calls = []
        turns = [
            {"type": "function_call_output", "call_id": call["id"], "output": json.dumps(feedback)}
            for call in self.pending
        ]
        content = [{"type": "input_text", "text": text}]
        for name, data in images:
            content.extend(
                [
                    {"type": "input_text", "text": name},
                    # `detail` is optional to OpenAI but required by compatible servers
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": "data:image/png;base64," + data,
                    },
                ]
            )
        turn = {"role": "user", "content": content}
        turns.append(turn)
        payload = {
            "model": self.model,
            "instructions": system,
            "input": self.history + turns,
            "tools": [dict(type="function", strict=False, **tool) for tool in tools],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_output_tokens": self.max_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if self.reasoning:
            payload["reasoning"] = {"effort": self.reasoning}
        response = self._post("/responses", payload)
        turns.extend(response.get("output", []))
        for item in response.get("output", []):
            if item["type"] == "function_call":
                try:
                    args = json.loads(item["arguments"])
                except (ValueError, TypeError):
                    args = None
                calls.append({"id": item["call_id"], "name": item["name"], "args": args})
        return turns, calls, response.get("usage", {})

    def _anthropic(self, text, images, feedback, system, tools):
        """One Messages API request; returns (new turns, tool calls, usage)."""
        content = [
            {"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(feedback)}
            for call in self.pending
        ]
        content.append({"type": "text", "text": text})
        for name, data in images:
            content.extend(
                [
                    {"type": "text", "text": name},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": data},
                    },
                ]
            )
        turn = {"role": "user", "content": content}
        payload = {
            "model": self.model,
            "system": system,
            "messages": [*self.history, turn],
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
                for tool in tools
            ],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "max_tokens": self.max_tokens,
            "cache_control": {"type": "ephemeral"},
        }
        if self.reasoning:
            payload.update(thinking={"type": "adaptive"}, output_config={"effort": self.reasoning})
        response = self._post("/messages", payload)
        native = response.get("content", [])
        turns = [turn, {"role": "assistant", "content": native}]
        calls = [
            {"id": block["id"], "name": block["name"], "args": block["input"]}
            for block in native
            if block["type"] == "tool_use"
        ]
        return turns, calls, response.get("usage", {})

    def _gemini(self, text, images, feedback, system, tools):
        """One generateContent request; returns (new turns, tool calls, usage)."""
        calls = []
        parts = []
        for call in self.pending:
            result = {"name": call["name"], "response": feedback}
            if call.get("id"):
                result["id"] = call["id"]
            parts.append({"functionResponse": result})
        parts.append({"text": text})
        for name, data in images:
            parts.extend([{"text": name}, {"inlineData": {"mimeType": "image/png", "data": data}}])
        turn = {"role": "user", "parts": parts}
        declarations = copy.deepcopy(tools)
        for declaration in declarations:
            # Gemini's Schema subset does not accept additionalProperties.
            declaration["parameters"].pop("additionalProperties", None)
            for prop in declaration["parameters"]["properties"].values():
                prop.pop("additionalProperties", None)
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [*self.history, turn],
            "tools": [{"functionDeclarations": declarations}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }
        if self.reasoning:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": self.reasoning}
        response = self._post(
            "/models/" + urllib.parse.quote(self.model, safe="") + ":generateContent", payload
        )
        candidates = response.get("candidates", [])
        turns = [turn]
        if candidates and candidates[0].get("content"):
            native = candidates[0]["content"]
            turns.append(native)
            calls = [
                {
                    "id": part["functionCall"].get("id"),
                    "name": part["functionCall"]["name"],
                    "args": part["functionCall"].get("args", {}),
                }
                for part in native["parts"]
                if "functionCall" in part
            ]
        else:
            # Record a blocked reply as a model turn so user and model turns still alternate.
            turns.append({"role": "model", "parts": [{"text": "(no response)"}]})
        return turns, calls, response.get("usageMetadata", {})
