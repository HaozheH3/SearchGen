"""
OpenAI-compatible model wrapper used by the SearchGen reasoner.

This file intentionally provides:
1) A lightweight LLM client with retry logic.
2) JSON parsing helpers that tolerate fenced/partial outputs.
3) Backward-compatible FrontierModel methods used by pipeline/evaluation scripts.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # pragma: no cover — urllib3 ships with requests; be defensive
    pass

try:
    from ..media.vlm_payload import (
        DEFAULT_MAX_TOTAL_PIXELS,
        build_refinement_user_content,
        build_refinement_user_content_interleaved,
        build_user_content_text_then_image_parts,
        local_file_to_vlm_data_url,
        local_image_file_is_openable,
    )
except ImportError:  # pragma: no cover
    DEFAULT_MAX_TOTAL_PIXELS = 768 * 768  # type: ignore[assignment, misc]
    build_refinement_user_content = None  # type: ignore[assignment, misc]
    build_refinement_user_content_interleaved = None  # type: ignore[assignment, misc]
    build_user_content_text_then_image_parts = None  # type: ignore[assignment, misc]
    local_file_to_vlm_data_url = None  # type: ignore[assignment, misc]
    local_image_file_is_openable = None  # type: ignore[assignment, misc]


def _load_dotenv_if_available() -> None:
    """
    Load environment variables from ``.env`` when python-dotenv is available.
    We keep this optional to avoid hard dependency issues in environments without python-dotenv.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        # If python-dotenv isn't installed, just rely on process env.
        return


# Load environment variables early for CLI use.
_load_dotenv_if_available()


def _first_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


TASK_A_SYSTEM_PROMPT = (
    "You are an expert at predicting where image generation models (like DALL-E, Midjourney, "
    "Flux, Stable Diffusion) will FAIL or produce inaccurate results. Your job is NOT to find "
    'things that "would benefit" from references — almost everything would. Your job is to '
    "identify genuine KNOWLEDGE GAPS where the model will likely hallucinate, render "
    "incorrectly, or fail entirely WITHOUT external grounding (text or image reference).\n\n"
    "You have deep understanding of:\n"
    "- What these models learned from training data (generally pre-2024)\n"
    "- Common failure modes: faces of less-famous people, recent events, niche cultural items, "
    "precise symbols, text rendering, specific product designs, etc.\n"
    "- What these models handle WELL: generic scenes, common objects, well-known landmarks, "
    "popular celebrities, standard artistic styles, common animals, basic compositions."
)

TASK_A_USER_TEMPLATE = """Analyze this image generation prompt for visual reference candidates:
"{user_prompt}"

## VISUAL REFERENCE TAXONOMY

Identify ALL entities, concepts, and elements that COULD benefit from visual references. Use this taxonomy to classify each candidate:

### Category Definitions:

**TK-R (Temporal Knowledge - Recent):** Entities that emerged or significantly changed AFTER ~mid-2024.
**TK-C (Temporal Knowledge - Current):** Entities requiring real-time or very recent information.
**EIK (Entity & IP Knowledge):** Specific named entities where visual accuracy matters.
**CSV (Concept & Symbol Visualization):** Abstract concepts, flags, diagrams, symbols requiring precise representation.
**FHA (Factual/Historical Accuracy):** Historical scenes, scientific illustrations, period-accurate depictions.
**CS (Cultural Specificity):** Culture-specific visual elements requiring authentic representation.
**VUR (Visual/UI/UX/Rendering):** UI elements, app layouts, specific visual styles, rendering techniques.
**THG (Text/Typography/Graphic):** Specific text rendering, typography, graphic design elements.

## COMPREHENSIVE ANALYSIS APPROACH

- Be INCLUSIVE, not conservative.
- List all entities/concepts that may benefit from references.
- Estimate severity honestly: critical|important|moderate|minor.
- Suggest concrete search queries and search type.

## OUTPUT FORMAT

Provide a JSON object with this exact structure:
{{
  "analysis_reasoning": "2-3 sentences",
  "knowledge_gaps": [
    {{
      "entity": "name",
      "category": "TK-R|TK-C|EIK|CSV|FHA|CS|VUR|THG|OTHER",
      "severity": "critical|important|moderate|minor",
      "reasoning": "why reference helps",
      "suggested_search": "query",
      "search_type": "image|web"
    }}
  ],
  "search_queries": ["ordered by priority, max 3-5"],
  "search_justification": "what searches provide"
}}

IMPORTANT: Respond in Chinese when the user prompt is Chinese, otherwise in English for clarity."""

# Tail of the analyze-prompt user message (after the quoted user prompt and optional hint block).
# Kept separate so hint mode inserts a plug-and-play section without changing this text.
_ANALYZE_PROMPT_USER_MESSAGE_TAIL = """## VISUAL REFERENCE TAXONOMY

Identify ALL entities, concepts, and elements that COULD benefit from visual references. Use this taxonomy to classify each candidate:

### Category Definitions:

**TK-R (Temporal Knowledge - Recent):** Entities that emerged or significantly changed AFTER ~mid-2024.
**TK-C (Temporal Knowledge - Current):** Entities requiring real-time or very recent information.
**EIK (Entity & IP Knowledge):** Specific named entities where visual accuracy matters.
**CSV (Concept & Symbol Visualization):** Abstract concepts, flags, diagrams, symbols requiring precise representation.
**FHA (Factual/Historical Accuracy):** Historical scenes, scientific illustrations, period-accurate depictions.
**CS (Cultural Specificity):** Culture-specific visual elements requiring authentic representation.
**VUR (Visual/UI/UX/Rendering):** UI elements, app layouts, specific visual styles, rendering techniques.
**THG (Text/Typography/Graphic):** Specific text rendering, typography, graphic design elements.

## COMPREHENSIVE ANALYSIS APPROACH

- Be INCLUSIVE, not conservative.
- List all entities/concepts that may benefit from references.
- Estimate severity honestly: critical|important|moderate|minor.
- Suggest concrete search queries and search type.

## OUTPUT FORMAT

Provide a JSON object with this exact structure:
{
  "analysis_reasoning": "2-3 sentences",
  "knowledge_gaps": [
    {
      "entity": "name",
      "category": "TK-R|TK-C|EIK|CSV|FHA|CS|VUR|THG|OTHER",
      "severity": "critical|important|moderate|minor",
      "reasoning": "why reference helps",
      "suggested_search": "query",
      "search_type": "image|web"
    }
  ],
  "search_queries": ["ordered by priority, max 3-5"],
  "search_justification": "what searches provide"
}

IMPORTANT: Respond in Chinese when the user prompt is Chinese, otherwise in English for clarity."""

# User-message tail for prompt analysis (same JSON schema as legacy ``_ANALYZE_PROMPT_USER_MESSAGE_TAIL``).
_ANALYZE_PROMPT_USER_MESSAGE_TAIL_COMPARE = """## KNOWLEDGE GAP TAXONOMY

Identify knowledge gaps—information the model may lack or get wrong—where **web** search (facts, text, conventions) and/or **image** search (appearance) can ground the answer. Classify each gap with this taxonomy:

### Category Definitions:

**TK-R (Temporal Knowledge - Recent):** Entities that emerged or significantly changed AFTER ~mid-2024.
**TK-C (Temporal Knowledge - Current):** Entities requiring real-time or very recent information.
**EIK (Entity & IP Knowledge):** Specific named entities where visual accuracy matters.
**CSV (Concept & Symbol Visualization):** Abstract concepts, flags, diagrams, symbols requiring precise representation.
**FHA (Factual/Historical Accuracy):** Historical scenes, scientific illustrations, period-accurate depictions.
**CS (Cultural Specificity):** Culture-specific visual elements requiring authentic representation.
**VUR (Visual/UI/UX/Rendering):** UI elements, app layouts, specific visual styles, rendering techniques.
**THG (Text/Typography/Graphic):** Specific text rendering, typography, graphic design elements.

## COMPREHENSIVE ANALYSIS APPROACH

- Be INCLUSIVE, not conservative.
- List gaps (and affected entities/concepts) plainly.
- Estimate severity honestly: critical|important|moderate|minor.
- Suggest concrete search queries and **search_type** (`web` or `image`); use **web** when the gap is mainly factual/textual and **image** when it is mainly visual.
- When **search_type** is `web`, phrase **suggested_search** as a short natural-language *question or descriptive lookup* (what / how / why / definitions / standards / authoritative descriptions)—not a bag of keywords and not written to harvest URLs. For `image`, keep **suggested_search** focused on visible subject matter.

## OUTPUT FORMAT

Provide a JSON object with this exact structure:
{
  "analysis_reasoning": "2-3 sentences",
  "knowledge_gaps": [
    {
      "entity": "name",
      "category": "TK-R|TK-C|EIK|CSV|FHA|CS|VUR|THG|OTHER",
      "severity": "critical|important|moderate|minor",
      "reasoning": "why reference helps",
      "suggested_search": "query",
      "search_type": "image|web"
    }
  ],
  "search_queries": ["ordered by priority, max 3-5"],
  "search_justification": "what searches provide"
}

IMPORTANT: Respond in Chinese when the user prompt is Chinese, otherwise in English for clarity."""


def normalize_prompt_analysis_hint_context(
    hint_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Build a minimal hint dict from dataset fields ``text_search_queries`` and
    ``why_text_search_needed``. Returns None when there is nothing usable.
    """
    if not hint_context:
        return None
    out: Dict[str, Any] = {}
    tsq = hint_context.get("text_search_queries")
    if isinstance(tsq, list):
        cleaned: List[Dict[str, str]] = []
        for item in tsq:
            if not isinstance(item, dict):
                continue
            q = str(item.get("query", "")).strip()
            purpose = str(item.get("purpose", "")).strip()
            if q or purpose:
                cleaned.append({"query": q, "purpose": purpose})
        if cleaned:
            out["text_search_queries"] = cleaned
    why = hint_context.get("why_text_search_needed")
    if isinstance(why, str) and why.strip():
        out["why_text_search_needed"] = why.strip()
    nsi = hint_context.get("no_search_improvement_context")
    if isinstance(nsi, str) and nsi.strip():
        out["no_search_improvement_context"] = nsi.strip()
    return out or None


def _analysis_search_mode_preface_from_env() -> str:
    """
    Optional extra user-message text for prompt analysis (``analyze_prompt``).

    When ``SEARCHGEN_ANALYSIS_SEARCH_MODE=web_primary``, steer the model toward web-grounding
    first and treat image search as optional—without baking dataset-specific strings into
    callers.

    When ``SEARCHGEN_ANALYSIS_SEARCH_MODE=balanced``, ask for a balanced mix of web and image
    steps, slightly favoring web. Unset / other values: no extra text (legacy behavior).
    """
    mode = (os.environ.get("SEARCHGEN_ANALYSIS_SEARCH_MODE") or "").strip().lower()
    if mode == "web_primary":
        return (
            "## Search modality preference\n\n"
            "For this run, prioritize **web** lookups for grounding (facts, terminology, standards, "
            "recent events, and textual constraints). Use **image** search only when the gap is "
            "predominantly about precise visible appearance that web text alone cannot resolve. "
            "When both modalities could help, prefer **web-first** planning and treat image search "
            "as optional supplemental grounding—not the default.\n\n"
        )
    if mode == "balanced":
        return (
            "## Search modality preference\n\n"
            "Propose a **balanced** execution plan that uses both **web** and **image** search when "
            "useful: **web** for facts, terminology, standards, dates, and textual constraints; "
            "**image** when precise visible appearance, layout, or hard-to-describe visual detail "
            "matters. When either modality could address the gap, **lean slightly toward web-first**: "
            "treat image search as complementing web grounding rather than replacing it.\n\n"
        )
    return ""


def format_prompt_analysis_hint_block(hint_context: Optional[Dict[str, Any]]) -> str:
    """Return optional dataset hint as compact markdown (no JSON blob)."""
    normalized = normalize_prompt_analysis_hint_context(hint_context)
    if not normalized:
        return ""
    chunks: List[str] = ["## Dataset hint"]
    why = normalized.get("why_text_search_needed")
    if isinstance(why, str) and why.strip():
        chunks.append(why.strip())
    nsi = normalized.get("no_search_improvement_context")
    if isinstance(nsi, str) and nsi.strip():
        chunks.append(nsi.strip())
    tsq = normalized.get("text_search_queries")
    if isinstance(tsq, list) and tsq:
        bullets: List[str] = []
        for item in tsq:
            if not isinstance(item, dict):
                continue
            q = str(item.get("query", "")).strip()
            purpose = str(item.get("purpose", "")).strip()
            if q and purpose:
                bullets.append(f"- *{q}* — {purpose}")
            elif q:
                bullets.append(f"- *{q}*")
            elif purpose:
                bullets.append(f"- {purpose}")
        if bullets:
            chunks.append(
                "Suggested text-search directions from the record (adapt as needed):\n"
                + "\n".join(bullets)
            )
    return "\n\n".join(chunks) + "\n"


_SCRUB_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _scrub_web_evidence_text(text: str) -> str:
    """Strip URLs and collapse blank lines from bundled web-evidence text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    out = _SCRUB_URL_RE.sub("", text)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def _normalize_borrow_row(row: Any) -> Optional[Dict[str, Any]]:
    """Ensure ``index`` like ``visual-1`` / ``text-1`` exists; keep legacy ``image_index`` fields."""
    if not isinstance(row, dict):
        return None
    d = dict(row)
    if not str(d.get("index", "")).strip():
        if d.get("image_index") is not None:
            try:
                d["index"] = f"visual-{int(d['image_index'])}"
            except (TypeError, ValueError):
                pass
        elif d.get("reference_index") is not None:
            try:
                d["index"] = f"visual-{int(d['reference_index'])}"
            except (TypeError, ValueError):
                pass
    return d


def _borrow_slot_kind_num(row: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """Parse ``index`` (``visual-2``, ``text-1``) or legacy image/reference index → (kind, n)."""
    idx = row.get("index")
    if isinstance(idx, str):
        s = idx.strip().lower()
        for prefix, kind in (("visual-", "visual"), ("image-", "visual"), ("text-", "text")):
            if s.startswith(prefix):
                rest = s[len(prefix) :]
                if rest.isdigit():
                    return kind, int(rest)
                break
    for key in ("image_index", "reference_index"):
        if row.get(key) is not None:
            try:
                return "visual", int(row[key])
            except (TypeError, ValueError):
                pass
    return "", None


def _filter_borrow_rows(
    rows: List[Dict[str, Any]],
    n_visual_slots: int,
    n_text_slots: int,
) -> List[Dict[str, Any]]:
    """Drop ``visual-*`` / ``text-*`` rows that do not match evidence actually included in the prompt."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        kind, num = _borrow_slot_kind_num(row)
        if kind == "visual":
            if n_visual_slots <= 0 or num is None or not (1 <= num <= n_visual_slots):
                continue
        elif kind == "text":
            if n_text_slots <= 0 or num is None or not (1 <= num <= n_text_slots):
                continue
        else:
            continue
        out.append(row)
    return out


def _refinement_response_format_snippet(
    n_visual_slots: int, n_text_slots: int
) -> Tuple[str, str, str]:
    """
    Build (borrow_from_references JSON lines, reasoning example phrase, combine_strategy example phrase)
    for the refinement prompt — avoids showing ``visual-*`` / ``text-*`` examples when that evidence is absent.
    """
    if n_visual_slots > 0 and n_text_slots > 0:
        borrow = """  "borrow_from_references": [
    {{
      "index": "visual-1",
      "used": true,
      "reference_focus": "what you take from Reference Image 1"
    }},
    {{
      "index": "text-1",
      "used": true,
      "reference_focus": "facts or wording from Text Reference 1 (only from excerpts above)"
    }}
  ],"""
        reason = "how both textual and visual references were incorporated"
        combine = "how to blend textual and visual references"
    elif n_visual_slots > 0:
        borrow = """  "borrow_from_references": [
    {{
      "index": "visual-1",
      "used": true,
      "reference_focus": "what you take from Reference Image 1"
    }}
  ],"""
        reason = "how the image reference(s) were incorporated"
        combine = "how the image reference(s) inform the final prompt"
    elif n_text_slots > 0:
        borrow = """  "borrow_from_references": [
    {{
      "index": "text-1",
      "used": true,
      "reference_focus": "facts or wording from Text Reference 1 (only from excerpts above)"
    }}
  ],"""
        reason = "how the web search evidence was incorporated"
        combine = "how web evidence constrains the final prompt"
    else:
        borrow = """  "borrow_from_references": [],"""
        reason = "brief note (no image or web excerpts were provided in this run)"
        combine = "brief note if no references apply"
    return borrow, reason, combine


def build_analyze_prompt_user_message(
    user_prompt: str, hint_context: Optional[Dict[str, Any]] = None
) -> str:
    """User message for ``analyze_prompt`` (knowledge-gap tail + optional dataset hint)."""
    head = (
        "Analyze this image generation prompt for knowledge gaps:\n"
        f'"{user_prompt}"'
    )
    mode_preface = _analysis_search_mode_preface_from_env()
    tail = mode_preface + _ANALYZE_PROMPT_USER_MESSAGE_TAIL_COMPARE
    hint_body = format_prompt_analysis_hint_block(hint_context)
    if hint_body:
        return head + "\n\n" + hint_body + "\n\n" + tail
    return head + "\n\n" + tail


TASK_B_SYSTEM_PROMPT = (
    "You are an expert at analyzing candidate reference images for image generation tasks. "
    "Your goal is to identify what visual or factual knowledge an AIGC model would be missing, "
    "then select the single image that best fills those knowledge gaps. Follow the required two-step "
    "protocol: first identify the key knowledge gaps, then choose the image that most directly provides "
    "the needed visual grounding. Return structured JSON only and do not add markdown or extra text."
)

TASK_B_USER_TEMPLATE = """Select the best reference image for this image generation task.

User Request:
"{user_prompt}"

Search Query Used:
"{query}"

Candidate Images (you can see them visually):
{candidate_images}

STEP 1: IDENTIFY KNOWLEDGE GAPS
Analyze what knowledge gaps an AIGC model might have when generating this image:

1. **Niche Entities** — Specific products, brands, or objects not in training data
   - Example: "Tesla Cybertruck" (relatively new, specific design)
   - Example: "specific celebrity or public figure"

2. **Identity & Appearance** — Specific people, characters, or visual identities
   - Example: "What does this specific person look like?"
   - Example: "What is the exact design of this product?"

3. **Visual References** — Specific styles, compositions, or visual techniques
   - Example: "What does this architectural style look like?"
   - Example: "What is the lighting mood in this setting?"

4. **Cultural/Contextual Details** — Culture-specific or context-specific information
   - Example: "What does this location look like?"
   - Example: "What is the visual style of this era?"

Write 2-3 sentences identifying the main knowledge gaps that an AIGC model would need to address.

STEP 2: SELECT IMAGE THAT FILLS GAPS
After identifying the knowledge gaps, select the image that best provides visual grounding:

1. Which image most directly addresses the identified knowledge gaps?
2. Which image provides the clearest visual reference for what the model needs to know?
3. Consider: specificity, clarity, and relevance to the knowledge gaps (NOT user request matching)

Provide a JSON object with this structure:

{{
  "identified_knowledge_gaps": "Description of what AIGC model needs to know (2-3 sentences)",
  "selected_index": 0-{max_index},
  "gap_filling_reasoning": "How this image fills the identified gaps"
}}

Important: First identify the knowledge gaps, then select the image that best fills them.

{language_instruction}"""

TASK_C_SYSTEM_PROMPT = (
    "You are an expert at creating detailed, specific prompts for text-to-image generation models "
    "when the request is grounded by reference **images** and—when provided—**text evidence** from web "
    "search (facts, labels, wording, or constraints). Your task is to deeply analyze the user's intent "
    "and produce a comprehensive specification that guides high-quality, accurate generation: use images "
    "for appearance, layout, and style, and use text snippets wherever they resolve factual or textual "
    "ambiguity the images alone do not cover."
)

# TASK_C_USER_TEMPLATE = """You are given multiple reference images and a user's request.

# Your task has TWO coupled outputs:
# 1) Reference mapping plan: explicitly specify what to refer to in each visual reference image.
# 2) Prompt enhancement: polish and expand the initial user prompt to improve generation quality while preserving user intent.

# Create a detailed, high-quality prompt that leverages all useful references while maintaining creative freedom and ensuring accurate rendering of key visual elements.

# Original User Request: "{user_prompt}"

# Visual Reference Candidates:
# {visual_reference_candidates}

# Reference Images (you can see them visually):
# {reference_images}

# RESPONSE FORMAT (JSON only, no markdown):
# {{
#   "visual_planning": "2-3 sentences",
#   "reasoning": "how references were incorporated",
#   "borrow_from_references": [
#     {{
#       "image_index": 1,
#       "used": true,
#       "reference_focus": "specific region/aspect"
#     }}
#   ],
#   "combine_strategy": "how to blend references",
#   "create_new": ["new element 1", "new element 2"],
#   "refined_prompt": "Detailed prompt with explicit selected-reference guidance"
# }}

# CRITICAL: Return ONLY valid JSON. No markdown, no extra text.
# {language_instruction}"""


@dataclass
class UsageStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_prompt_cost: float = 0.0
    total_completion_cost: float = 0.0
    total_cost: float = 0.0
    request_count: int = 0

    def add(self, usage: Dict[str, Any], cost_info: Dict[str, Any]) -> None:
        self.total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        self.total_prompt_cost += float(cost_info.get("prompt_cost", 0.0) or 0.0)
        self.total_completion_cost += float(cost_info.get("completion_cost", 0.0) or 0.0)
        self.total_cost += float(cost_info.get("total_cost", 0.0) or 0.0)
        self.request_count += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "request_count": self.request_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_prompt_cost": self.total_prompt_cost,
            "total_completion_cost": self.total_completion_cost,
            "total_cost": self.total_cost,
        }


@dataclass
class LLMClient:
    model_name: str
    default_timeout: int = 180
    usage_stats_print_frequency: int = 0
    usage_stats_printer: Callable[[str], None] = print
    aggregate_usage_stats: UsageStats = field(default_factory=UsageStats)

    def __post_init__(self) -> None:
        base_url = _first_env(
            "SEARCHGEN_CHAT_BASE_URL",
            "FRONTIER_LLM_API_URL",
            "LLM_API_URL",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
        )
        if not base_url:
            raise RuntimeError(
                "Missing chat endpoint. Set SEARCHGEN_CHAT_BASE_URL or pass --chat-base-url."
            )
        self.base_url = base_url.rstrip("/")
        api_format = (_first_env("SEARCHGEN_CHAT_API_FORMAT", "LLM_API_FORMAT") or "openai").strip().lower()
        if api_format not in {"openai", "vllm"}:
            raise RuntimeError("SEARCHGEN_CHAT_API_FORMAT must be 'openai' or 'vllm'")
        self.api_format = api_format
        self.api_key = _first_env("SEARCHGEN_CHAT_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY") or ""

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        # vLLM deployments often run without auth. Only attach bearer token when available.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _openai_chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _vllm_chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def resolved_chat_url(self) -> str:
        """HTTP URL used for chat requests after ``api_format`` resolution."""
        if self.api_format == "vllm":
            return self._vllm_chat_completions_url()
        return self._openai_chat_completions_url()

    @staticmethod
    def _normalize_usage_stats(usage: Any) -> Dict[str, Any]:
        if not isinstance(usage, dict):
            return {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    @staticmethod
    def _normalize_cost_info(cost_info: Any, fallback_total_cost: Any = None) -> Dict[str, Any]:
        normalized = cost_info if isinstance(cost_info, dict) else {}
        total_cost = normalized.get("total_cost", fallback_total_cost)
        return {
            "prompt_cost": float(normalized.get("prompt_cost", 0.0) or 0.0),
            "completion_cost": float(normalized.get("completion_cost", 0.0) or 0.0),
            "total_cost": float(total_cost or 0.0),
        }

    def _record_usage_stats(self, usage: Any, cost_info: Any, *, context: str) -> None:
        normalized_usage = self._normalize_usage_stats(usage)
        normalized_cost_info = self._normalize_cost_info(cost_info)
        if not normalized_usage and not normalized_cost_info.get("total_cost"):
            return

        self.aggregate_usage_stats.add(normalized_usage, normalized_cost_info)
        freq = int(self.usage_stats_print_frequency or 0)
        if freq > 0 and self.aggregate_usage_stats.request_count % freq == 0:
            summary = self.aggregate_usage_stats.as_dict()
            self.usage_stats_printer(
                "[FrontierModel usage] "
                f"context={context} requests={summary['request_count']} "
                f"prompt_tokens={summary['total_prompt_tokens']} "
                f"completion_tokens={summary['total_completion_tokens']} "
                f"total_tokens={summary['total_tokens']} "
                f"prompt_cost={summary['total_prompt_cost']:.8f} "
                f"completion_cost={summary['total_completion_cost']:.8f} "
                f"total_cost={summary['total_cost']:.8f}"
            )

    def get_aggregate_usage_stats(self) -> Dict[str, Any]:
        return self.aggregate_usage_stats.as_dict()

    def reset_aggregate_usage_stats(self) -> None:
        self.aggregate_usage_stats = UsageStats()

    def _request_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        target_url = (
            self._vllm_chat_completions_url()
            if self.api_format == "vllm"
            else self._openai_chat_completions_url()
        )
        response = requests.post(
            target_url,
            headers=self._headers(),
            json=payload,
            timeout=timeout or self.default_timeout,
        )
        response.raise_for_status()
        raw = response.json()
        if isinstance(raw, dict):
            self._record_usage_stats(
                raw.get("usage"),
                raw.get("cost_info", {"total_cost": raw.get("total_cost")}),
                context=self.api_format,
            )
            if "usage" in raw:
                raw["usage"] = self._normalize_usage_stats(raw.get("usage"))
            raw["cost_info"] = self._normalize_cost_info(
                raw.get("cost_info"),
                fallback_total_cost=raw.get("total_cost"),
            )
        return raw

    def chat_with_retry(
        self,
        sys_prompt: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        max_retries: int = 2,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        last_error = None
        for attempt in range(max(1, max_retries) + 1):
            try:
                return self._request_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt >= max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        return {"_error": last_error or "chat_with_retry failed"}

    def multimodal_chat_multiimages_with_retry(
        self,
        sys_prompt: str,
        text_prompt: str,
        user_content: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
        max_retries: int = 2,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Keep text_prompt in signature for compatibility even when user_content already includes text.
        _ = text_prompt
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        last_error = None
        for attempt in range(max(1, max_retries) + 1):
            try:
                return self._request_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt >= max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        return {"_error": last_error or "multimodal_chat_multiimages_with_retry failed"}

    @staticmethod
    def _sniff_image_mime_from_bytes(header: bytes) -> Optional[str]:
        if len(header) < 12:
            return None
        if header[:4] == b"\x89PNG":
            return "image/png"
        if header[:2] == b"\xff\xd8":
            return "image/jpeg"
        if len(header) >= 6 and header[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return "image/webp"
        if header[:2] == b"BM":
            return "image/bmp"
        return None

    @staticmethod
    def _local_image_to_data_url(
        path: str,
        *,
        max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
        max_edge: int = 0,
    ) -> Optional[str]:
        """
        Build a ``data:`` URL for multimodal judge calls.

        Prefer Pillow → JPEG via ``vlm_image_payload`` (aspect-ratio pad to gateway limits,
        optional max-edge shrink, pixel budget, RGB). Fall back to raw base64 only when VLM helpers are unavailable.
        """
        p = Path(str(path or "").strip())
        if not p.exists() or not p.is_file():
            return None
        if local_file_to_vlm_data_url is not None:
            try:
                u = local_file_to_vlm_data_url(
                    str(p.resolve()),
                    max_total_pixels=int(max_total_pixels),
                    max_edge=int(max_edge),
                )
            except Exception:
                u = None
            if u:
                return u
            # Do not fall back to raw base64: corrupt/HTML-on-disk often passes weak sniffing
            # but still breaks the gateway PIL decode ("cannot identify image file").
            return None
        try:
            raw = p.read_bytes()
        except Exception:
            return None
        if not raw:
            return None
        mime = LLMClient._sniff_image_mime_from_bytes(raw[:32])
        if not mime:
            guessed, _ = mimetypes.guess_type(str(p))
            if guessed and guessed != "application/octet-stream":
                mime = guessed
        if not mime or mime == "application/octet-stream":
            return None
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def build_user_content_with_images(
        self,
        prompt: str,
        image_items: List[Any],
        *,
        max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
        max_edge: int = 0,
    ) -> List[Dict[str, Any]]:
        # Text first, then images (order matches Task C ``build_refinement_user_content``).
        if build_user_content_text_then_image_parts is not None:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            content.extend(
                build_user_content_text_then_image_parts(
                    image_items,
                    max_total_pixels=max_total_pixels,
                    max_edge=max_edge,
                )
            )
            return content

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for it in image_items or []:
            s = str(it or "").strip()
            if not s:
                continue
            sl = s.lower()
            if sl.startswith("http://") or sl.startswith("https://"):
                p: Dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": s},
                    "max_pixels": int(max_total_pixels),
                }
                content.append(p)
            else:
                data_url = self._local_image_to_data_url(s)
                if not data_url:
                    continue
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "max_pixels": int(max_total_pixels),
                    }
                )
        return content

    def extract_message_from_response(self, response: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        if not isinstance(response, dict):
            return False, None, "Response is not a dictionary"
        if "_error" in response:
            return False, None, str(response.get("_error"))
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return False, None, "Missing choices in model response"
        first = choices[0]
        if not isinstance(first, dict):
            return False, None, "Invalid first choice"
        message = first.get("message")
        if not isinstance(message, dict):
            return False, None, "Missing message in first choice"
        content = message.get("content")
        if isinstance(content, str):
            return True, content, None
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_chunks.append(str(item.get("text", "")))
            if text_chunks:
                return True, "\n".join(text_chunks), None
        return False, None, "Unable to extract textual message content"


def text_reasoner_routing_summary(model_name: str) -> Dict[str, str]:
    """
    Resolved text-LLM (reasoner) routing as used by ``LLMClient`` / ``FrontierModel`` from env.
    Intended for startup logs so runs show the actual HTTP target and format.
    """
    raw_llm_format = (_first_env("SEARCHGEN_CHAT_API_FORMAT", "LLM_API_FORMAT") or "").strip()
    client = LLMClient(model_name=model_name)
    chat_url = client.resolved_chat_url()
    auth = "Bearer" if client.api_key else "none"

    base_env_source = "SEARCHGEN_CHAT_BASE_URL" if _first_env("SEARCHGEN_CHAT_BASE_URL") else None
    if base_env_source is None and _first_env("FRONTIER_LLM_API_URL"):
        base_env_source = "FRONTIER_LLM_API_URL"
    if base_env_source is None and _first_env("LLM_API_URL"):
        base_env_source = "LLM_API_URL"
    if base_env_source is None and _first_env("OPENAI_BASE_URL"):
        base_env_source = "OPENAI_BASE_URL"
    if base_env_source is None and _first_env("OPENAI_API_BASE"):
        base_env_source = "OPENAI_API_BASE"
    if base_env_source is None:
        base_env_source = "(missing)"

    return {
        "Reasoner model id (request JSON `model`)": model_name,
        "Base URL env source": base_env_source,
        "LLM_API_FORMAT env": raw_llm_format or "(unset)",
        "Resolved API format": client.api_format,
        "Configured base URL": client.base_url,
        "HTTP chat endpoint (used)": chat_url,
        "Auth": auth,
    }


def _gap_reasoning_for_query_analysis(analysis: Dict[str, Any], query: str) -> str:
    """Best-effort gap ``reasoning`` for a search query (kept in sync with ``pipeline._gap_reasoning_for_query``)."""
    key = str(query or "").strip().lower()
    if not key or not isinstance(analysis, dict):
        return ""
    for pool_name in ("execution_candidates", "knowledge_gaps", "visual_reference_candidates"):
        pool = analysis.get(pool_name)
        if not isinstance(pool, list):
            continue
        for ent in pool:
            if not isinstance(ent, dict):
                continue
            sq = str(ent.get("suggested_search", "")).strip().lower()
            en = str(ent.get("entity", "")).strip().lower()
            if sq == key or en == key:
                return str(ent.get("reasoning", "") or "").strip()
    return ""


class FrontierModel:
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_ANALYZE_MAX_TOKENS = 8192
    DEFAULT_REFINE_MAX_TOKENS = 8192

    def __init__(
        self,
        model_name: str = "qwen3.5-plus",
        temperature: float = 0.2,
        api_format: Optional[str] = None,
        usage_stats_print_frequency: int = 0,
        max_total_pixels: int = DEFAULT_MAX_TOTAL_PIXELS,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.api_format = api_format
        # FRONTIER_VLM_MAX_PIXELS overrides the default pixel budget (useful for self-served vLLM
        # servers with limited context length). Explicit caller-provided value always wins.
        _env_pixels = os.getenv("FRONTIER_VLM_MAX_PIXELS")
        if _env_pixels and max_total_pixels == DEFAULT_MAX_TOTAL_PIXELS:
            max_total_pixels = int(_env_pixels)
        self.max_total_pixels = int(max_total_pixels)
        self.client = LLMClient(
            model_name=model_name,
            usage_stats_print_frequency=usage_stats_print_frequency,
        )
        if api_format:
            normalized = api_format.strip().lower()
            if normalized not in {"openai", "vllm"}:
                raise ValueError("api_format must be 'openai' or 'vllm'")
            self.client.api_format = normalized

    def get_aggregate_usage_stats(self) -> Dict[str, Any]:
        return self.client.get_aggregate_usage_stats()

    def reset_aggregate_usage_stats(self) -> None:
        self.client.reset_aggregate_usage_stats()

    @staticmethod
    def _extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None

        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        cleaned = text
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            obj = json.loads(cleaned)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end == -1:
            return None
        try:
            obj = json.loads(text[start:end])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def _normalize_search_type(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"image", "web"}:
            return raw
        return "image"

    @staticmethod
    def _heuristic_entities(user_prompt: str) -> List[Dict[str, Any]]:
        prompt = user_prompt.strip()
        if not prompt:
            return []

        candidates: List[str] = []
        seen: set = set()

        # Quoted spans are often explicit entities.
        for q in re.findall(r'"([^"]+)"', prompt):
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                candidates.append(q)

        # Capitalized phrases (simple English heuristic).
        for phrase in re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", prompt):
            phrase = phrase.strip()
            if len(phrase) < 3:
                continue
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                candidates.append(phrase)

        # Comma-delimited key phrases fallback.
        if not candidates:
            for part in re.split(r"[，,;/]", prompt):
                part = part.strip()
                if len(part) >= 4 and part.lower() not in seen:
                    seen.add(part.lower())
                    candidates.append(part)

        entities: List[Dict[str, Any]] = []
        for item in candidates[:5]:
            entities.append(
                {
                    "entity": item,
                    "entity_type": "object",
                    "necessity_level": "high",
                    "reason": "Likely visually specific token in prompt.",
                    "suggested_search": item,
                    "search_type": "image",
                }
            )
        return entities

    def analyze_prompt(
        self, user_prompt: str, hint_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are an expert at predicting where image generation models (like DALL-E, Midjourney, "
            "Flux, Stable Diffusion) will FAIL or produce inaccurate results. Your job is NOT to find "
            'things that "would benefit" from references — almost everything would. Your job is to '
            "identify genuine KNOWLEDGE GAPS where the model will likely hallucinate, render "
            "incorrectly, or fail entirely WITHOUT external grounding (web facts or reference imagery).\n\n"
            "You have deep understanding of:\n"
            "- What these models learned from training data (generally pre-2024)\n"
            "- Common failure modes: faces of less-famous people, recent events, niche cultural items, "
            "precise symbols, text rendering, specific product designs, etc.\n"
            "- What these models handle WELL: generic scenes, common objects, well-known landmarks, "
            "popular celebrities, standard artistic styles, common animals, basic compositions.\n\n"
            "CRITICAL LANGUAGE RULE: You MUST respond in the SAME language as the user prompt. "
            "If the prompt is in English, ALL output fields (analysis_reasoning, knowledge_gaps, "
            "search_queries, search_justification) MUST be in English. If Chinese, use Chinese. "
            "Never mix languages or default to Chinese for English prompts."
        )
        if hint_context is not None:
            system_prompt += (
                "\n\n"
                "CRITICAL PRIVACY RULE: The user message contains calibration data. "
                "You MUST NOT use any of the following FORBIDDEN PHRASES "
                "in ANY output field (analysis_reasoning, knowledge_gaps, search_justification, etc.):\n"
                '  - "baseline quality" / "baseline score" / "基线质量" / "基线分数"\n'
                '  - "strong baseline" / "weak baseline" / "高基线" / "低基线"\n'
                '  - "without search" / "无搜索" / "no-search baseline"\n'
                '  - "with search and references" / "有搜索"\n'
                '  - Any number followed by "/100" (like "64.3/100" or "50/100")\n'
                '  - "calibration hint" / "校准提示" / "校准数据"\n'
                '  - Any paraphrase of the above (e.g. "the provided score", "the baseline suggests")\n\n'
                "Use the calibration data ONLY to silently adjust gap severities. "
                "When the baseline is strong, downgrade gap severities "
                "(critical→important, important→moderate) — do NOT eliminate gaps. "
                "Violating this rule corrupts the dataset and invalidates your output."
            )
        prompt = build_analyze_prompt_user_message(user_prompt, hint_context)
        raw_llm_input: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.temperature),
            "max_tokens": int(self.DEFAULT_ANALYZE_MAX_TOKENS),
        }
        api_resp = self.client.chat_with_retry(
            sys_prompt=system_prompt,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.DEFAULT_ANALYZE_MAX_TOKENS,
            max_retries=2,
        )
        ok, response_text, err = self.client.extract_message_from_response(api_resp)
        if not ok:
            return {
                "success": False,
                "analysis": None,
                "raw_response": response_text,
                "raw_llm_input": raw_llm_input,
                "error": err or "Failed to extract message from model response",
            }
        parsed = self._extract_and_parse_json(response_text or "")
        if not isinstance(parsed, dict):
            return {
                "success": False,
                "analysis": None,
                "raw_response": response_text,
                "raw_llm_input": raw_llm_input,
                "error": "Failed to parse analysis JSON",
            }

        candidates = parsed.get("knowledge_gaps")
        if not isinstance(candidates, list):
            candidates = parsed.get("visual_reference_candidates")
        if not isinstance(candidates, list):
            candidates = []
        normalized_candidates: List[Dict[str, Any]] = []
        for ent in candidates:
            if not isinstance(ent, dict):
                continue
            normalized_candidates.append(
                {
                    "entity": str(ent.get("entity", "")).strip(),
                    "category": str(ent.get("category", "OTHER")).strip().upper() or "OTHER",
                    "severity": str(ent.get("severity", "minor")).strip().lower(),
                    "reasoning": str(ent.get("reasoning", "")).strip(),
                    "suggested_search": str(ent.get("suggested_search", "")).strip(),
                    "search_type": self._normalize_search_type(ent.get("search_type")),
                }
            )

        search_queries = parsed.get("search_queries")
        if not isinstance(search_queries, list):
            search_queries = []

        limited_candidates = [
            cand for cand in normalized_candidates
            if str(cand.get("severity", "")).strip().lower() in {"critical", "important"}
        ]

        normalized_queries = [str(q).strip() for q in search_queries if str(q).strip()]
        if not normalized_queries:
            normalized_queries = self.build_execution_search_queries(
                {"execution_candidates": limited_candidates}
            )

        needs_search = bool(parsed.get("needs_search"))
        if normalized_queries:
            needs_search = True

        analysis = {
            "analysis_reasoning": str(parsed.get("analysis_reasoning", "")).strip(),
            "visual_reference_candidates": normalized_candidates,
            "search_queries": normalized_queries,
            "search_justification": str(parsed.get("search_justification", "")).strip(),
            "knowledge_gaps": normalized_candidates,
            "needs_search": needs_search,
            "execution_candidates": limited_candidates,
        }
        analysis["execution_search_plan"] = self.build_execution_search_plan(analysis, max_steps=3)
        return {
            "success": True,
            "analysis": analysis,
            "raw_response": response_text,
            "raw_llm_input": raw_llm_input,
        }

    @staticmethod
    def _severity_rank(severity: str) -> int:
        order = {"critical": 3, "important": 2, "moderate": 1, "minor": 0}
        return order.get(str(severity or "").strip().lower(), 0)

    @staticmethod
    def _search_type_rank(search_type: str) -> int:
        return 1 if str(search_type or "").strip().lower() == "image" else 0

    def build_execution_search_plan(
        self, analysis: Dict[str, Any], max_steps: int = 3
    ) -> List[Dict[str, str]]:
        """
        Build up to ``max_steps`` search calls: each {"query", "search_type"}.

        Source priority (first non-empty wins):
          1. critical/important ``execution_candidates``
          2. critical/important ``knowledge_gaps``  (fallback — some S1 schemas
             populate knowledge_gaps but never set execution_candidates)
          3. plain ``search_queries``  (last resort — preserves search intent
             even when no severity-tagged entities are present)

        Rationale: search must run whenever S1 surfaced critical/important
        entities. Earlier behaviour read only ``execution_candidates`` with no
        fallback, so S1 outputs that carried critical gaps in ``knowledge_gaps``
        (without ``execution_candidates``) silently skipped search. See
        PROTOCOLS_20K_DATASET_SCHEMA.md §10.3 (D5 HIGHLIGHT).

        Dedupes by query (case-insensitive): keeps higher severity, then image over web.
        Final order: severity desc, then image over web.
        """
        def _rows_from(entities: Any) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if not isinstance(entities, list):
                return out
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                sev = str(ent.get("severity", "")).strip().lower()
                if sev not in {"critical", "important"}:
                    continue
                q = str(ent.get("suggested_search", "")).strip() or str(ent.get("entity", "")).strip()
                if not q:
                    continue
                st = self._normalize_search_type(ent.get("search_type"))
                out.append({"query": q, "search_type": st, "severity": sev})
            return out

        # Priority 1: execution_candidates; Priority 2: knowledge_gaps fallback.
        rows: List[Dict[str, Any]] = _rows_from(analysis.get("execution_candidates"))
        if not rows:
            rows = _rows_from(analysis.get("knowledge_gaps"))
        # Priority 3: plain search_queries (no severity/type metadata available).
        if not rows:
            for q in (analysis.get("search_queries") or []):
                q = str(q).strip()
                if q:
                    rows.append({"query": q, "search_type": self._normalize_search_type(None), "severity": "critical"})

        best: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            key = str(r["query"]).lower()
            if key not in best:
                best[key] = r
                continue
            cur = best[key]
            cand_tuple = (self._severity_rank(r["severity"]), self._search_type_rank(r["search_type"]))
            cur_tuple = (self._severity_rank(cur["severity"]), self._search_type_rank(cur["search_type"]))
            if cand_tuple > cur_tuple:
                best[key] = r

        merged = list(best.values())
        merged.sort(
            key=lambda x: (self._severity_rank(x["severity"]), self._search_type_rank(x["search_type"])),
            reverse=True,
        )
        return [
            {"query": str(x["query"]).strip(), "search_type": str(x["search_type"]).strip().lower()}
            for x in merged[:max_steps]
            if str(x.get("query", "")).strip() and str(x.get("search_type", "")).strip().lower() in {"image", "web"}
        ]

    def build_execution_search_queries(self, analysis: Dict[str, Any], max_queries: int = 3) -> List[str]:
        plan = analysis.get("execution_search_plan")
        if isinstance(plan, list) and plan:
            return [str(s.get("query", "")).strip() for s in plan if isinstance(s, dict)][:max_queries]

        entities = (
            analysis.get("execution_candidates")
            or analysis.get("visual_reference_candidates")
            or analysis.get("knowledge_gaps")
            or []
        )
        queries: List[str] = []
        if isinstance(entities, list):
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                severity = str(ent.get("severity", "")).strip().lower()
                if severity not in {"critical", "important", "moderate"}:
                    continue
                q = str(ent.get("suggested_search", "")).strip() or str(ent.get("entity", "")).strip()
                if q and q not in queries:
                    queries.append(q)
                if len(queries) >= max_queries:
                    break
        if not queries:
            src = analysis.get("search_queries")
            if isinstance(src, list):
                for q in src:
                    q = str(q).strip()
                    if q and q not in queries:
                        queries.append(q)
                    if len(queries) >= max_queries:
                        break
        return queries

    def select_reference_image(
        self,
        user_prompt: str,
        analysis: Dict[str, Any],
        query: str,
        candidate_images: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        valid = []
        for idx, img in enumerate(candidate_images):
            if not isinstance(img, dict):
                continue
            url = str(img.get("url", "")).strip()
            if not url:
                continue
            if local_image_file_is_openable is not None:
                lp0 = str(img.get("local_path", "") or "").strip()
                http_ok = url.lower().startswith("http://") or url.lower().startswith("https://")
                viable = http_ok
                if lp0:
                    p0 = Path(lp0).expanduser()
                    if p0.is_file() and local_image_file_is_openable(p0):
                        viable = True
                if not viable:
                    continue
            valid.append((idx, img))
        if not valid:
            return {
                "success": False,
                "error": "No candidate image with valid URL or decodable local file for VLM",
            }

        candidate_image_mappings = []
        candidate_lines: List[str] = []
        for display_idx, (idx, img) in enumerate(valid, start=1):
            title = str(img.get("title", "")).strip()
            source = str(img.get("source", "")).strip()
            selection_input_source = "original_url"
            candidate_image_mappings.append(
                {
                    "index": display_idx - 1,
                    "title": title,
                    "url": str(img.get("url", "")),
                    "local_path": img.get("local_path"),
                    "selection_input_source": selection_input_source,
                    "selection_input_size_bytes": None,
                }
            )
            candidate_lines.append(
                f"Image {display_idx}:\n"
                f"  Title: {title}\n"
                f"  Source: {source}\n"
                f"  Selection Input: {selection_input_source}"
            )

        language_instruction = (
            "IMPORTANT: Respond in Chinese to match the user's original language."
            if any("\u4e00" <= ch <= "\u9fff" for ch in user_prompt)
            else "IMPORTANT: Respond in English for clarity and consistency."
        )

        prompt = TASK_B_USER_TEMPLATE.format(
            user_prompt=user_prompt,
            query=query,
            candidate_images="\n\n".join(candidate_lines),
            max_index=max(0, len(valid) - 1),
            language_instruction=language_instruction,
        )
        # One slot per candidate: prefer on-disk file from the pipeline, else hotlink the URL
        # (``vlm_image_payload`` does not download remote URLs).
        image_items: List[Any] = []
        for _, img in valid:
            lp = str((img or {}).get("local_path", "") or "").strip()
            u = str((img or {}).get("url", "") or "").strip()
            if lp:
                pth = Path(lp).expanduser()
                if pth.is_file():
                    image_items.append(str(pth))
                    continue
            if u and (u.lower().startswith("http://") or u.lower().startswith("https://")):
                image_items.append(u)
                continue
            if lp:
                image_items.append(str(Path(lp).expanduser()))
        user_content = self.client.build_user_content_with_images(
            prompt=prompt,
            image_items=image_items,
            max_total_pixels=self.max_total_pixels,
        )
        api_resp = self.client.multimodal_chat_multiimages_with_retry(
            sys_prompt=TASK_B_SYSTEM_PROMPT,
            text_prompt=prompt,
            user_content=user_content,
            temperature=self.temperature,
            max_tokens=1200,
            max_retries=2,
        )
        ok, response_text, err = self.client.extract_message_from_response(api_resp)
        if not ok:
            return {
                "success": False,
                "error": f"api_call_failed: {err}",
                "selection_prompt": prompt,
                "selection_response": response_text,
                "candidate_image_mappings": candidate_image_mappings,
            }
        parsed = self._extract_and_parse_json(response_text or "")
        if not isinstance(parsed, dict):
            return {
                "success": False,
                "error": "response_not_parseable",
                "selection_prompt": prompt,
                "selection_response": response_text,
                "candidate_image_mappings": candidate_image_mappings,
            }
        try:
            selected_zero_based = int(parsed.get("selected_index"))
        except (TypeError, ValueError):
            return {
                "success": False,
                "error": "selected_index_missing_or_not_int",
                "selection_prompt": prompt,
                "selection_response": response_text,
                "candidate_image_mappings": candidate_image_mappings,
            }
        if not (0 <= selected_zero_based < len(valid)):
            return {
                "success": False,
                "error": f"selected_index_out_of_range: got {selected_zero_based}, valid 0..{len(valid)-1}",
                "selection_prompt": prompt,
                "selection_response": response_text,
                "candidate_image_mappings": candidate_image_mappings,
            }

        best_idx, best_img = valid[selected_zero_based]
        identified_knowledge_gaps = parsed.get(
            "identified_knowledge_gaps", analysis.get("missing_elements", [])
        )
        selection_reasoning = str(parsed.get("gap_filling_reasoning", "")).strip()

        return {
            "success": True,
            "selected_image": best_img,
            "selection_reasoning": selection_reasoning,
            "identified_knowledge_gaps": identified_knowledge_gaps,
            "visual_description": f"Reference image: {str(best_img.get('title', '')).strip()} from {str(best_img.get('source', '')).strip()}",
            "selection_prompt": prompt,
            "selection_response": response_text,
            "retry_count": 0,
            "selection_input_source": "original_url",
            "selection_input_size_bytes": None,
            "candidate_image_mappings": candidate_image_mappings,
        }

    def refine_prompt(
        self,
        user_prompt: str,
        analysis: Dict[str, Any],
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Refine prompt with optional search-backed references."""
        system_prompt = TASK_C_SYSTEM_PROMPT

        image_rows: List[Dict[str, Any]] = []
        web_rows: List[Dict[str, Any]] = []
        for ref in search_results or []:
            if not isinstance(ref, dict):
                continue
            t = str(ref.get("type", "")).strip().lower()
            if t == "web":
                web_rows.append(ref)
            elif t == "image":
                image_rows.append(ref)
            elif ref.get("imageUrl") or ref.get("url"):
                image_rows.append(ref)
            else:
                web_rows.append(ref)

        ref_pairs: List[Tuple[str, Dict[str, Any]]] = []
        vis_slot = 0
        for ref in image_rows:
            if not isinstance(ref, dict):
                continue
            vis_slot += 1
            title = ref.get("title", "Unknown")
            query = str(ref.get("query", "") or "")
            purpose = _gap_reasoning_for_query_analysis(analysis, query)
            if not purpose.strip():
                purpose = str(ref.get("selection_reasoning", "") or "").strip()
            if not purpose.strip():
                purpose = "(not specified)"
            chunk = (
                f"Reference Image {vis_slot} (from query: '{query}' ; why this search: {purpose}) <image>\n"
                f"- Title: {title}\n"
            )
            ref_pairs.append((chunk, ref))

        ref_line_blocks = [chunk for chunk, _ in ref_pairs]
        references_text = (
            "\n\n".join(ref_line_blocks) if ref_line_blocks else "No reference images provided."
        )

        web_by_query: Dict[str, Dict[str, Any]] = {}
        for r in web_rows:
            qk = str(r.get("query", "") or "").strip()
            if qk and qk not in web_by_query:
                web_by_query[qk] = r

        text_section_lines: List[str] = []
        for q_idx, (qkey, sample) in enumerate(web_by_query.items(), start=1):
            meta = (
                sample.get("text_reference_metadata")
                if isinstance(sample.get("text_reference_metadata"), dict)
                else {}
            )
            purpose = str(meta.get("why_text_search", "") or "").strip()
            ref_text = _scrub_web_evidence_text(
                str(sample.get("text_reference", "") or "").strip()
            )
            text_section_lines.append(
                f"Text Reference {q_idx} (web search query: \"{qkey}\")\n"
                f"- Purpose (why this search): {purpose or '(not specified)'}\n"
                f"- Top-3 searched results excerpts (snippet text only; page titles and URLs omitted):\n"
                f"{ref_text or '(no snippets returned)'}"
            )
        if text_section_lines:
            text_refs_block = (
                "### Text search evidence (web)\n" + "\n\n".join(text_section_lines)
            )
        else:
            text_refs_block = (
                "### Text search evidence (web)\nNo web search evidence provided."
            )

        n_visual_slots = len(ref_pairs)
        n_text_slots = len(web_by_query)
        borrow_fmt, reason_eg, combine_eg = _refinement_response_format_snippet(
            n_visual_slots, n_text_slots
        )

        if not ref_pairs and not web_by_query:
            visual_refs_block = """### Visual Reference Candidates:
  No specific visual reference candidates identified."""
        elif not ref_pairs and web_by_query:
            visual_refs_block = (
                "Reference Images (numbered sequentially as Reference Image 1, 2, …):\n"
                "No reference images provided.\n"
            )
        else:
            visual_refs_block = (
                "Reference Images (numbered sequentially as Reference Image 1, 2, …; "
                "same order as any images shown to you):\n"
                f"{references_text}"
            )

        language_instruction = (
            "The refined prompt should be written in Chinese to match the user's original language"
            if any("\u4e00" <= ch <= "\u9fff" for ch in user_prompt)
            else "The refined prompt should be written in English for best generation results"
        )
        prompt_head = f"""You are given reference images (optional), web search evidence (optional), and a user's request.

Your task has TWO coupled outputs:
1) Reference mapping plan: for each **image** reference, what to use visually; for each **web** block, which facts or constraints you take from the query purpose and the top result summaries (do not invent numbers or claims beyond the evidence).
2) Prompt enhancement: polish and expand the initial user prompt to improve generation quality while preserving user intent.

Create a detailed, high-quality prompt that leverages all useful evidence while maintaining creative freedom and ensuring accurate rendering of key visual elements.

Original User Request: "{user_prompt}"

{text_refs_block}

"""
        ref_intro = (
            "Reference Images (numbered sequentially as Reference Image 1, 2, …; "
            "each raster attached in the multimodal message appears **immediately after** the line "
            "that contains ``<image>`` for that reference, in the same order).\n"
        )
        prompt_tail = f"""

CITATION RULE: In ``refined_prompt``, immediately after every borrowed visual element from a **Reference Image n** above, append exactly (refer to Reference Image n) or (参考图n) using the same ``n`` as in that heading, choose the right language based on the language of the original prompt.

Field notes (not JSON keys): ``create_new`` lists compositional or stylistic elements that are **not** copied directly from a reference but should still appear so the final image stays coherent (e.g. invented background detail, spacing, lighting invented beyond what references show).

UNDERSPECIFIED PROMPTS: When the original user request is vague, underspecified, or requires implicit visual planning (e.g., "a girl", "write a spring couplet", "a sonnet", "a scene about hope"), you MUST enrich it with concrete implicit details rather than passing it through as-is. Infer reasonable defaults for unspecified aspects: subject appearance, setting, composition, lighting, mood, style, and spatial layout. For text-heavy requests (couplets, poems, calligraphy, typography), explicitly plan the visual layout, typography, decorative elements, and material context. The refined_prompt must be self-contained and renderable without the generator needing to guess what to draw.

RESPONSE FORMAT (JSON only, no markdown):
{{
  "visual_planning": "2-3 sentences",
  "reasoning": "{reason_eg}",
{borrow_fmt}
  "combine_strategy": "{combine_eg}",
  "create_new": ["short label for a non-borrowed element needed for coherence", "…"],
  "refined_prompt": "Detailed prompt with explicit selected-reference guidance"
}}

Rules: emit ``visual-<n>`` only when Reference Image n exists above (same n); emit ``text-<m>`` only when Text Reference m exists above. Do **not** invent ``visual-*`` or ``text-*`` rows for missing sections—use ``[]`` or omit those indices. Legacy ``image_index`` is accepted but prefer ``index``.

CRITICAL: Return ONLY valid JSON. No markdown, no extra text.
{language_instruction}"""
        if ref_pairs:
            preamble = prompt_head + ref_intro
            prompt = preamble + "".join(chunk for chunk, _ in ref_pairs) + prompt_tail
        else:
            prompt = prompt_head + visual_refs_block + prompt_tail

        ref_dicts: List[Dict[str, Any]] = [r for r in image_rows if isinstance(r, dict)]
        vlm_info: Dict[str, Any] = {
            "mode": "text_only",
            "n_image_parts": 0,
            "image_payloads": [],
        }
        can_interleave = bool(ref_pairs and build_refinement_user_content_interleaved)
        use_multimodal = bool(ref_dicts and (can_interleave or build_refinement_user_content))
        if use_multimodal and can_interleave:
            bres = build_refinement_user_content_interleaved(
                prompt_head + ref_intro,
                ref_pairs,
                prompt_tail,
                max_total_pixels=self.max_total_pixels,
                max_edge=0,
            )
            n_img = len(bres.image_metas)
            if n_img > 0 and len(bres.user_content) > 1:
                vlm_info = {
                    "mode": "multimodal_interleaved",
                    "n_image_parts": n_img,
                    "max_total_pixels": self.max_total_pixels,
                    "image_payloads": [
                        {
                            "strategy": m.strategy,
                            "source": m.source,
                            "byte_length": m.byte_length,
                            "had_resize": m.had_resize,
                        }
                        for m in bres.image_metas
                    ],
                }
                api_resp = self.client.multimodal_chat_multiimages_with_retry(
                    sys_prompt=system_prompt,
                    text_prompt=prompt,
                    user_content=bres.user_content,
                    temperature=self.temperature,
                    max_tokens=self.DEFAULT_REFINE_MAX_TOKENS,
                    max_retries=2,
                )
            elif build_refinement_user_content is not None:
                bres = build_refinement_user_content(
                    prompt,
                    ref_dicts,
                    max_total_pixels=self.max_total_pixels,
                    max_edge=0,
                )
                n_img = len(bres.image_metas)
                if n_img > 0 and len(bres.user_content) > 1:
                    vlm_info = {
                        "mode": "multimodal",
                        "n_image_parts": n_img,
                        "max_total_pixels": self.max_total_pixels,
                        "image_payloads": [
                            {
                                "strategy": m.strategy,
                                "source": m.source,
                                "byte_length": m.byte_length,
                                "had_resize": m.had_resize,
                            }
                            for m in bres.image_metas
                        ],
                    }
                    api_resp = self.client.multimodal_chat_multiimages_with_retry(
                        sys_prompt=system_prompt,
                        text_prompt=prompt,
                        user_content=bres.user_content,
                        temperature=self.temperature,
                        max_tokens=self.DEFAULT_REFINE_MAX_TOKENS,
                        max_retries=2,
                    )
                else:
                    vlm_info["mode"] = "text_only"
                    vlm_info["n_image_parts"] = 0
                    vlm_info["reason"] = "no_resolvable_image_payloads"
                    api_resp = self.client.chat_with_retry(
                        sys_prompt=system_prompt,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.DEFAULT_REFINE_MAX_TOKENS,
                        max_retries=2,
                    )
            else:
                vlm_info["mode"] = "text_only"
                vlm_info["n_image_parts"] = 0
                vlm_info["reason"] = "no_interleave_builder"
                api_resp = self.client.chat_with_retry(
                    sys_prompt=system_prompt,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.DEFAULT_REFINE_MAX_TOKENS,
                    max_retries=2,
                )
        elif use_multimodal and build_refinement_user_content is not None:
            bres = build_refinement_user_content(
                prompt,
                ref_dicts,
                max_total_pixels=self.max_total_pixels,
                max_edge=0,
            )
            n_img = len(bres.image_metas)
            if n_img > 0 and len(bres.user_content) > 1:
                vlm_info = {
                    "mode": "multimodal",
                    "n_image_parts": n_img,
                    "max_total_pixels": self.max_total_pixels,
                    "image_payloads": [
                        {
                            "strategy": m.strategy,
                            "source": m.source,
                            "byte_length": m.byte_length,
                            "had_resize": m.had_resize,
                        }
                        for m in bres.image_metas
                    ],
                }
                api_resp = self.client.multimodal_chat_multiimages_with_retry(
                    sys_prompt=system_prompt,
                    text_prompt=prompt,
                    user_content=bres.user_content,
                    temperature=self.temperature,
                    max_tokens=self.DEFAULT_REFINE_MAX_TOKENS,
                    max_retries=2,
                )
            else:
                vlm_info["mode"] = "text_only"
                vlm_info["n_image_parts"] = 0
                vlm_info["reason"] = "no_resolvable_image_payloads"
                api_resp = self.client.chat_with_retry(
                    sys_prompt=system_prompt,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.DEFAULT_REFINE_MAX_TOKENS,
                    max_retries=2,
                )
        else:
            vlm_info["mode"] = "text_only"
            vlm_info["reason"] = (
                "no_reference_dicts" if not ref_dicts else "vlm_image_payload_unavailable"
            )
            api_resp = self.client.chat_with_retry(
                sys_prompt=system_prompt,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.DEFAULT_REFINE_MAX_TOKENS,
                max_retries=2,
            )
        ok, response_text, err = self.client.extract_message_from_response(api_resp)
        parsed = self._extract_and_parse_json(response_text or "") if ok else None
        if isinstance(parsed, dict) and str(parsed.get("refined_prompt", "")).strip():
            borr_raw = parsed.get("borrow_from_references", [])
            borr_norm: List[Dict[str, Any]] = []
            if isinstance(borr_raw, list):
                for x in borr_raw:
                    nb = _normalize_borrow_row(x)
                    if nb is not None:
                        borr_norm.append(nb)
            borr_filt = _filter_borrow_rows(borr_norm, n_visual_slots, n_text_slots)
            return {
                "success": True,
                "visual_planning": str(parsed.get("visual_planning", "")).strip(),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
                "borrow_from_references": borr_filt,
                "combine_strategy": str(parsed.get("combine_strategy", "")).strip(),
                "create_new": parsed.get("create_new", []),
                "refined_prompt": str(parsed.get("refined_prompt")).strip(),
                "raw_input_prompt": prompt,
                "raw_output_response": response_text,
                "refinement_vlm_submission": vlm_info,
            }
        if not ok:
            # API call itself failed — do NOT pretend the original prompt is a valid refinement
            return {
                "success": False,
                "error": err or "API call failed",
                "visual_planning": "",
                "reasoning": "",
                "borrow_from_references": [],
                "combine_strategy": "",
                "create_new": [],
                "refined_prompt": "",
                "raw_input_prompt": prompt,
                "raw_output_response": err or "",
                "refinement_vlm_submission": vlm_info,
            }
        # Parse failed but API succeeded — fallback to original prompt (acceptable)
        return {
            "success": True,
            "visual_planning": "",
            "reasoning": "Fallback: original prompt kept because refinement parse failed.",
            "borrow_from_references": [],
            "combine_strategy": "",
            "create_new": [],
            "refined_prompt": user_prompt,
            "raw_input_prompt": prompt,
            "raw_output_response": response_text,
            "refinement_vlm_submission": vlm_info,
        }

    def refine_prompt_with_reference(
        self,
        user_prompt: str,
        analysis: Dict[str, Any],
        selected_image: Dict[str, Any],
        visual_description: str = "",
        web_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Task C fallback: one selected reference, same path as :meth:`refine_prompt` (multimodal if possible)."""
        if not isinstance(selected_image, dict):
            return {
                "success": False,
                "error": "refine_prompt_with_reference: selected_image must be a dict",
                "refined_prompt": user_prompt,
            }
        merged: Dict[str, Any] = dict(selected_image)
        if visual_description and not str(merged.get("selection_reasoning", "")).strip():
            merged["selection_reasoning"] = visual_description
        web_rows = [
            r
            for r in (web_results or [])
            if isinstance(r, dict) and str(r.get("type", "")).strip().lower() == "web"
        ]
        merged_search = web_rows + [merged]
        return self.refine_prompt(user_prompt, analysis, search_results=merged_search)

    def refine_prompt_with_multiple_references(
        self,
        user_prompt: str,
        analysis: Dict[str, Any],
        reference_images: List[Dict[str, Any]],
        web_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not reference_images:
            return self.refine_prompt(user_prompt=user_prompt, analysis=analysis, search_results=None)

        selected_indices: List[int] = []
        borrowed_traits: List[str] = []
        selected_images: List[Dict[str, Any]] = []
        for idx, ref in enumerate(reference_images, start=1):
            if idx > 3:
                break
            selected = ref.get("selected_image", {}) if isinstance(ref, dict) else {}
            if isinstance(selected, dict):
                img_pack = dict(selected)
                if isinstance(ref, dict):
                    qq = str(ref.get("query", "") or "").strip()
                    if qq:
                        img_pack["query"] = qq
                    ssr = str(ref.get("selection_reasoning", "") or "").strip()
                    if ssr and not str(img_pack.get("selection_reasoning", "") or "").strip():
                        img_pack["selection_reasoning"] = ssr
                selected_images.append(img_pack)
            title = str(selected.get("title", "")).strip()
            if title:
                borrowed_traits.append(title)
            selected_indices.append(idx)

        web_only = [
            r
            for r in (web_results or [])
            if isinstance(r, dict) and str(r.get("type", "")).strip().lower() == "web"
        ]
        result = self.refine_prompt(
            user_prompt=user_prompt,
            analysis=analysis,
            search_results=web_only + selected_images,
        )
        if result.get("success"):
            result["selected_reference_indices"] = selected_indices
            borr = result.get("borrow_from_references")
            model_borrow: List[Dict[str, Any]] = []
            if isinstance(borr, list):
                for item in borr:
                    if not isinstance(item, dict):
                        continue
                    row = dict(item)
                    kind, n = _borrow_slot_kind_num(row)
                    if kind == "visual" and n is not None and n >= 1:
                        if row.get("reference_index") is None:
                            row["reference_index"] = n
                        if row.get("image_index") is None:
                            row["image_index"] = n
                        if n <= len(borrowed_traits) and not row.get("borrowed_traits"):
                            row["borrowed_traits"] = [borrowed_traits[n - 1]]
                    nb = _normalize_borrow_row(row)
                    if nb is not None:
                        model_borrow.append(nb)
            if model_borrow:
                result["borrow_from_references"] = model_borrow
            else:
                result["borrow_from_references"] = [
                    {
                        "index": f"visual-{i}",
                        "reference_index": i,
                        "image_index": i,
                        "borrowed_traits": (
                            [borrowed_traits[i - 1]] if i - 1 < len(borrowed_traits) else []
                        ),
                    }
                    for i in selected_indices
                ]
            if not str(result.get("combine_strategy", "")).strip():
                result["combine_strategy"] = (
                    "Blend the selected references by using their core visual traits as "
                    "grounding while keeping the user's original scene intact."
                )
            if not result.get("create_new"):
                result["create_new"] = [
                    "new scene composition",
                    "new lighting treatment",
                    "new background details",
                ]
        return result
