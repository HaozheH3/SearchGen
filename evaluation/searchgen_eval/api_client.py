from __future__ import annotations

import os
import time
from typing import Any

import requests


class APIClient:
    def __init__(self, endpoint: str | None, api_key: str | None, model: str, timeout: float = 180):
        self.endpoint = endpoint or os.environ.get("SEARCHGEN_EVAL_API_URL") or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("SEARCHGEN_EVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.timeout = timeout
        if self.endpoint and not self.endpoint.rstrip("/").endswith("chat/completions"):
            self.endpoint = self.endpoint.rstrip("/") + "/chat/completions"

    def chat(self, system_prompt: str, user_content: list[dict], temperature: float = .2,
             max_tokens: int = 8000, retries: int = 2) -> tuple[str, Any]:
        if not self.endpoint or not self.api_key:
            raise RuntimeError("API endpoint/key missing; set SEARCHGEN_EVAL_API_URL and SEARCHGEN_EVAL_API_KEY")
        payload = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]}
        error = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(self.endpoint, headers={"Authorization": f"Bearer {self.api_key}"},
                                         json=payload, timeout=self.timeout)
                response.raise_for_status()
                raw = response.json()
                return raw["choices"][0]["message"]["content"], raw
            except Exception as exc:
                error = exc
                if attempt < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"API request failed: {error}")
