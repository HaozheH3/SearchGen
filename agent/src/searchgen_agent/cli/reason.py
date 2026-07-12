"""Run the Agentic Reasoner with Search Tools on JSON or JSONL prompts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ..io import load_rows
from ..reasoner.row_state import enqueue_plan, histogram
from ..reasoner.stagewise_runner import V3StagewiseRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prompt dataset (.json or .jsonl)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root")
    parser.add_argument("--dataset-name", default=None, help="Portable dataset label; defaults to input stem")
    parser.add_argument("--lane-name", default="agentic_reasoner", help="Output lane label")
    parser.add_argument("--model", required=True, help="Multimodal model for S2/S3")
    parser.add_argument("--model-s1", default=None, help="Optional text model for S1")
    parser.add_argument("--chat-base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--search-api-url", default=None, help="Search provider endpoint")
    parser.add_argument(
        "--search-signer",
        default=None,
        help="Trusted callback as package.module:function; signer loads its own credentials",
    )
    parser.add_argument("--mode", choices=("full", "stop_s2"), default="full")
    parser.add_argument("--workers-s1", type=int, default=4)
    parser.add_argument("--workers-s1d1-search", type=int, default=8)
    parser.add_argument("--workers-s1d1-download", type=int, default=8)
    parser.add_argument("--workers-s2", type=int, default=2)
    parser.add_argument("--workers-s3", type=int, default=2)
    parser.add_argument("--max-downloaded-images-per-query", type=int, default=3)
    parser.add_argument("--max-search-results", type=int, default=10)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.input)
    if args.row_index is not None and not 1 <= args.row_index <= len(rows):
        raise SystemExit(f"--row-index must be between 1 and {len(rows)}")

    if args.chat_base_url:
        os.environ["SEARCHGEN_CHAT_BASE_URL"] = args.chat_base_url
    if args.search_api_url:
        os.environ["SEARCHGEN_SEARCH_API_URL"] = args.search_api_url
    if args.search_signer:
        os.environ["SEARCHGEN_SEARCH_SIGNER"] = args.search_signer

    output_root = args.output_dir.expanduser().resolve()
    dataset_name = args.dataset_name or args.input.stem
    lane_root = output_root / args.lane_name
    row_dirs = [lane_root / f"eval_row_{index:03d}" for index in range(1, len(rows) + 1)]
    only = [args.row_index] if args.row_index is not None else None
    staging = output_root / "_work" / f"{dataset_name}_{args.lane_name}"
    runner = V3StagewiseRunner(
        rows=rows,
        gen_root=output_root,
        evalset=dataset_name,
        reasoner=args.lane_name,
        tree_subdir="",
        model_name=args.model,
        model_name_s1=args.model_s1,
        mode=args.mode,
        workers_s1=args.workers_s1,
        workers_s1d1=16,
        workers_s1d1_search=args.workers_s1d1_search,
        workers_s1d1_download=args.workers_s1d1_download,
        workers_s2=args.workers_s2,
        workers_s3=args.workers_s3,
        max_downloaded_images_per_query=args.max_downloaded_images_per_query,
        max_search_results=args.max_search_results,
        limit=args.limit,
        only_row_indices=only,
        staging_dir=staging,
        row_dirs=row_dirs,
    )

    states = runner.scan_all()
    summary = {
        "dataset": dataset_name,
        "lane": args.lane_name,
        "rows": len(rows),
        "states": histogram(states),
        "enqueue_plan": enqueue_plan(states, mode=args.mode),
        "output": str(lane_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if not args.yes and input("Proceed? (yes/no): ").strip().lower() not in {"yes", "y"}:
        print("cancelled")
        return 1
    result = asyncio.run(runner.run())
    print(json.dumps({"reasoner_run_summary": result}, ensure_ascii=False, indent=2))
    failed = int((result.get("counters") or {}).get("failed", 0) or 0)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
