"""
Atomic search handlers (sequential orchestration in the pipeline).

Each handler mirrors the prior in-pipeline I/O: SERP rows are tagged with
``query`` and ``type``; image selection output matches ``pipeline`` reference
bundles; text rows may carry ``text_reference`` and ``text_reference_metadata``.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)


def process_web_search_results_placeholder(_serp_rows: List[Dict[str, Any]]) -> None:
    """Reserved for richer text-SERP processing; intentionally empty for now."""


def run_text_search_for_query(
    query: str,
    *,
    search_client: Any,
    max_results: int,
    query_id: int,
    gap_reasoning: str = "",
) -> Dict[str, Any]:
    """
    Web SERP + minimal formatting. Returns same SERP shape as legacy ``_execute_search_plan``.

    Output keys:
      - ``serp_rows``: list of dicts (title, link, snippet, …) each with ``query``, ``type``=\"web\"
      - ``text_reference``: human-readable block (top 3 snippets only, no titles or URLs)
      - ``text_reference_metadata``: JSON-serializable dict (includes ``why_text_search``)
    """
    if not query.strip():
        return {"serp_rows": [], "text_reference": "", "text_reference_metadata": {}}

    result = search_client.search(
        query=query,
        search_type="web",
        num=max_results,
        query_id=query_id,
        max_downloaded_images=None,
    )

    serp_rows: List[Dict[str, Any]] = []
    if result.get("success"):
        for item in result["results"]:
            row = dict(item) if isinstance(item, dict) else {}
            row["query"] = query
            row["type"] = "web"
            serp_rows.append(row)

    process_web_search_results_placeholder(serp_rows)

    top = serp_rows[:3]
    lines: List[str] = []
    for i, r in enumerate(top, 1):
        snippet = str(r.get("snippet", "") or "").strip()
        if snippet:
            lines.append(f"{i}. {snippet}")
        else:
            lines.append(f"{i}. (no snippet text returned)")
    text_reference = "\n\n".join(lines) if lines else ""

    meta = {
        "query": query,
        "why_text_search": (gap_reasoning or "").strip(),
        "top_results": [
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "position": r.get("position"),
            }
            for r in top
        ],
    }

    for r in serp_rows:
        r["text_reference"] = text_reference
        r["text_reference_metadata"] = dict(meta)

    return {
        "serp_rows": serp_rows,
        "text_reference": text_reference,
        "text_reference_metadata": meta,
    }


def run_image_search_for_query(
    query: str,
    *,
    search_client: Any,
    frontier_model: Any,
    user_prompt: str,
    analysis: Dict[str, Any],
    max_results: int,
    query_id: int,
    query_index: int,
    max_downloaded_images: Optional[int],
    logger: Optional[logging.Logger] = None,
    skip_reference_selection: bool = False,
) -> Dict[str, Any]:
    """
    Image SERP → downloads (via client) → dedupe by title → ``select_reference_image``.

    Returns:
      - ``serp_rows``: tagged hit dicts (``query``, ``type``=\"image\")
      - ``selection``: same dict as legacy parallel helper, or ``None``
    """
    log = logger or _LOGGER

    if not query.strip():
        return {"serp_rows": [], "selection": None}

    result = search_client.search(
        query=query,
        search_type="image",
        num=max_results,
        query_id=query_id,
        max_downloaded_images=max_downloaded_images,
    )

    serp_rows: List[Dict[str, Any]] = []
    if result.get("success"):
        for item in result["results"]:
            row = dict(item) if isinstance(item, dict) else {}
            row["query"] = query
            row["type"] = "image"
            serp_rows.append(row)

    downloadable = [
        r for r in serp_rows if r.get("download_success", False) and r.get("imageUrl", "")
    ]
    if not downloadable:
        log.warning("No downloadable images for query: %r", query)
        return {"serp_rows": serp_rows, "selection": None}

    log.info("Query %s: %r — %d downloadable images", query_index + 1, query, len(downloadable))

    seen_titles: Dict[str, bool] = {}
    deduplicated: List[Dict[str, Any]] = []
    for hit in downloadable:
        title = hit.get("title", "").strip()
        normalized_title = " ".join(title.lower().split())
        if normalized_title and normalized_title not in seen_titles:
            seen_titles[normalized_title] = True
            deduplicated.append(
                {
                    **hit,
                    "url": hit.get("imageUrl", ""),
                    "success": True,
                    "query_index": query_index,
                }
            )

    log.info("Deduplicated to %d unique images for query: %r", len(deduplicated), query)
    if not deduplicated:
        return {"serp_rows": serp_rows, "selection": None}

    if skip_reference_selection:
        log.info("skip_reference_selection=1 — skipping select_reference_image (stop before s2)")
        return {"serp_rows": serp_rows, "selection": None}

    try:
        selection = frontier_model.select_reference_image(
            user_prompt,
            analysis,
            query,
            deduplicated,
        )
    except Exception as e:
        log.error("Image selection failed for query %r: %s", query, e)
        log.error(traceback.format_exc())
        return {"serp_rows": serp_rows, "selection": None}

    if not selection.get("success"):
        log.warning(
            "Failed to select image for query %r: %s",
            query,
            selection.get("error", "Unknown error"),
        )
        return {"serp_rows": serp_rows, "selection": None}

    selected_img = selection.get("selected_image", {})
    if not selected_img.get("url"):
        log.warning("Selected image has no valid URL for query: %r", query)
        return {"serp_rows": serp_rows, "selection": None}

    log.info(
        "Selected image for query %r: %s | url=%s | local_path=%s",
        query,
        selected_img.get("title", "Unknown"),
        selected_img.get("url", ""),
        selected_img.get("local_path", ""),
    )
    bundle = {
        "query": query,
        "query_index": query_index,
        "selected_image": selected_img,
        "selection_reasoning": selection.get("selection_reasoning", ""),
        "identified_knowledge_gaps": selection.get("identified_knowledge_gaps", ""),
        "visual_description": selection.get("visual_description", ""),
        "selection_prompt": selection.get("selection_prompt", ""),
        "selection_response": selection.get("selection_response", ""),
        "retry_count": selection.get("retry_count", 0),
        "selection_input_source": selection.get("selection_input_source", ""),
        "selection_input_size_bytes": selection.get("selection_input_size_bytes"),
        "candidate_image_mappings": selection.get(
            "candidate_image_mappings",
            [
                {
                    "index": idx,
                    "title": c.get("title", ""),
                    "url": c.get("url", ""),
                    "local_path": c.get("local_path", ""),
                    "selection_input_source": c.get("selection_input_source", ""),
                }
                for idx, c in enumerate(deduplicated)
            ],
        ),
    }
    return {"serp_rows": serp_rows, "selection": bundle}


def run_image_selection_from_s1d1_bucket(
    query: str,
    bucket_rows: List[Dict[str, Any]],
    *,
    frontier_model: Any,
    user_prompt: str,
    analysis: Dict[str, Any],
    query_index: int,
    logger: Optional[logging.Logger] = None,
    skip_reference_selection: bool = False,
) -> Dict[str, Any]:
    """
    Image **selection (s2)** using SERP-shaped rows from ``s1d1_search_results`` (corpus replay).

    Skips live image search / download; runs the same dedupe + ``select_reference_image`` path
    as ``run_image_search_for_query`` after SERP is populated.
    """
    log = logger or _LOGGER

    if not query.strip():
        return {"serp_rows": [], "selection": None}

    serp_rows: List[Dict[str, Any]] = []
    for r in bucket_rows:
        if not isinstance(r, dict):
            continue
        row = dict(r)
        row["query"] = query
        row["type"] = "image"
        iu = str(row.get("imageUrl") or row.get("url") or "").strip()
        row["imageUrl"] = iu
        row["url"] = str(row.get("url") or row.get("imageUrl") or iu).strip()
        lp = str(row.get("local_path") or "").strip()
        row["download_success"] = bool(row.get("downloaded")) and bool(lp) and bool(iu)
        serp_rows.append(row)

    def _http_u(r: Dict[str, Any]) -> str:
        return str(r.get("imageUrl") or r.get("url") or "").strip()

    downloadable = [r for r in serp_rows if r.get("download_success", False) and r.get("imageUrl", "")]
    if len(downloadable) < 2:
        log.warning("Query %r: only %d downloaded image(s), need >=2 for selection — skipping", query, len(downloadable))
        return {"serp_rows": serp_rows, "selection": None}
    pool = downloadable

    log.info(
        "Query %s: %r — %d candidate image(s) for selection (s1d1 replay; downloaded=%d)",
        query_index + 1,
        query,
        len(pool),
        len(downloadable),
    )

    seen_titles: Dict[str, bool] = {}
    deduplicated: List[Dict[str, Any]] = []
    for hit in pool:
        title = str(hit.get("title", "")).strip()
        normalized_title = " ".join(title.lower().split())
        if normalized_title and normalized_title not in seen_titles:
            seen_titles[normalized_title] = True
            deduplicated.append(
                {
                    **hit,
                    "url": hit.get("imageUrl", ""),
                    "success": True,
                    "query_index": query_index,
                }
            )

    log.info("Deduplicated to %d unique images for query: %r", len(deduplicated), query)
    if not deduplicated:
        return {"serp_rows": serp_rows, "selection": None}

    if skip_reference_selection:
        log.info("skip_reference_selection=1 — skipping select_reference_image (stop before s2)")
        return {"serp_rows": serp_rows, "selection": None}

    try:
        selection = frontier_model.select_reference_image(
            user_prompt,
            analysis,
            query,
            deduplicated,
        )
    except Exception as e:
        log.error("Image selection failed for query %r: %s", query, e)
        log.error(traceback.format_exc())
        return {"serp_rows": serp_rows, "selection": None}

    if not selection.get("success"):
        log.warning(
            "Failed to select image for query %r: %s",
            query,
            selection.get("error", "Unknown error"),
        )
        return {"serp_rows": serp_rows, "selection": None}

    selected_img = selection.get("selected_image", {})
    if not selected_img.get("url"):
        log.warning("Selected image has no valid URL for query: %r", query)
        return {"serp_rows": serp_rows, "selection": None}

    log.info(
        "Selected image for query %r: %s | url=%s | local_path=%s",
        query,
        selected_img.get("title", "Unknown"),
        selected_img.get("url", ""),
        selected_img.get("local_path", ""),
    )
    bundle = {
        "query": query,
        "query_index": query_index,
        "selected_image": selected_img,
        "selection_reasoning": selection.get("selection_reasoning", ""),
        "identified_knowledge_gaps": selection.get("identified_knowledge_gaps", ""),
        "visual_description": selection.get("visual_description", ""),
        "selection_prompt": selection.get("selection_prompt", ""),
        "selection_response": selection.get("selection_response", ""),
        "retry_count": selection.get("retry_count", 0),
        "selection_input_source": selection.get("selection_input_source", ""),
        "selection_input_size_bytes": selection.get("selection_input_size_bytes"),
        "candidate_image_mappings": selection.get(
            "candidate_image_mappings",
            [
                {
                    "index": idx,
                    "title": c.get("title", ""),
                    "url": c.get("url", ""),
                    "local_path": c.get("local_path", ""),
                    "selection_input_source": c.get("selection_input_source", ""),
                }
                for idx, c in enumerate(deduplicated)
            ],
        ),
    }
    return {"serp_rows": serp_rows, "selection": bundle}


# ─────────────────────── V3 split: search-only + download-only ───────────────────────
#
# These two helpers split the work performed by ``run_image_search_for_query`` into
# two stages so the V3 stagewise runner can schedule them in independent worker pools.
# Avoids blocking SERP capacity on slow image downloads (broken CDNs, 30s timeouts).
#
# S1D1a worker calls ``run_image_search_only_for_query`` per image query — returns SERP
#                  rows without downloading (10 raw image hits per query).
# S1D1b worker calls ``run_image_download_for_serp_rows`` per image query — downloads
#                  the previously-fetched SERP rows, dedupes by title, fills
#                  ``local_path`` / ``download_success`` fields.
# S2 worker is unchanged (still uses ``run_image_selection_from_s1d1_bucket`` over the
#            downloaded bucket).
#
# Neither helper invokes the LLM — selection (s2) remains in its own stage.


def run_image_search_only_for_query(
    query: str,
    *,
    search_client: Any,
    max_results: int,
    query_id: int,
    query_index: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Image SERP only — no downloads, no reference selection.

    Returns:
      - ``serp_rows``: list of dicts in the raw SERP shape (10 rows per query, no
        local_path / download_success), each tagged with ``query``, ``type='image'``,
        and ``query_index``.
    """
    log = logger or _LOGGER
    if not query.strip():
        return {"serp_rows": []}

    result = search_client.image_search(
        query=query,
        num=max_results,
        download_images=False,
        query_id=query_id,
    )

    serp_rows: List[Dict[str, Any]] = []
    if result.get("success"):
        for item in result.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["query"] = query
            row["type"] = "image"
            row["query_index"] = query_index
            serp_rows.append(row)

    log.info("[s1d1a] image SERP only: query=%r results=%d", query, len(serp_rows))
    return {"serp_rows": serp_rows}


def run_image_download_for_serp_rows(
    query: str,
    serp_rows: List[Dict[str, Any]],
    *,
    search_client: Any,
    query_id: int,
    query_index: int,
    max_downloaded_images: Optional[int],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Download images for a single image-query's pre-fetched SERP rows.

    Wraps ``SearchClient._download_images_parallel`` but preserves the full
    pre-fetched SERP list. Downloading is still deduped by title and capped by
    ``max_downloaded_images``; unattempted or failed rows remain in the returned
    list with ``download_success=False`` and ``local_path=None``. This keeps
    ``s1d1_search_results.json`` useful as a search log while S2 can still filter
    strictly to local downloaded images.
    """
    log = logger or _LOGGER
    if not serp_rows:
        return {"serp_rows": []}
    if not getattr(search_client, "output_dir", None):
        log.warning("[s1d1b] SearchClient has no output_dir; download skipped for query=%r", query)
        return {"serp_rows": serp_rows}

    downloaded = search_client._download_images_parallel(
        serp_rows,
        query_id=query_id,
        max_successful_downloads=max_downloaded_images,
    )

    def _image_url(row: Dict[str, Any]) -> str:
        return str(row.get("imageUrl") or row.get("url") or "").strip()

    downloaded_by_url: Dict[str, Dict[str, Any]] = {}
    for r in downloaded:
        if not isinstance(r, dict):
            continue
        u = _image_url(r)
        if u:
            downloaded_by_url[u] = r

    tagged: List[Dict[str, Any]] = []
    for r in serp_rows:
        if not isinstance(r, dict):
            continue
        row = dict(r)
        match = downloaded_by_url.get(_image_url(row))
        if match:
            row.update(match)
        success = bool(row.get("download_success")) and bool(row.get("local_path"))
        row["download_success"] = success
        row["downloaded"] = success
        if not success:
            row["local_path"] = None
        row["query"] = query
        row["type"] = "image"
        row["query_index"] = query_index
        tagged.append(row)

    n_success = sum(1 for r in tagged if r.get("download_success"))
    n_attempted = len(downloaded)
    log.info(
        "[s1d1b] downloaded %d/%d attempted images for query=%r; preserving %d/%d SERP rows",
        n_success, n_attempted, query, len(tagged), len(serp_rows),
    )
    return {"serp_rows": tagged}
