from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .api_client import APIClient
from .dataset import load_metadata
from .discovery import load_predictions
from .evaluator import build_request, evaluate, result_valid, RESULT


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SearchGen predictions with the canonical evaluation protocol.")
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--benchmark-root", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions-manifest", type=Path)
    group.add_argument("--images-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="doubao-seed-2.0-mini")
    p.add_argument("--endpoint")
    p.add_argument("--api-key")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--bench-id", action="append", default=[])
    p.add_argument("--generator")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--include-pp", action="store_true", help="Accepted for compatibility; PP is always enabled.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rows = load_metadata(args.metadata, args.benchmark_root)
    by_id = {row["sample_id"]: row for row in rows}
    predictions = load_predictions(args.predictions_manifest, args.images_dir)
    if args.bench_id:
        predictions = [p for p in predictions if p.get("bench_id") in set(args.bench_id)]
    if args.generator:
        predictions = [p for p in predictions if p.get("generator") == args.generator]
    errors = []
    jobs = []
    for pred in predictions:
        bench_id = pred.get("bench_id")
        if bench_id not in by_id:
            errors.append(f"unknown bench_id: {bench_id}")
            continue
        if not Path(pred["image_path"]).is_file():
            errors.append(f"missing prediction: {pred['image_path']}")
            continue
        jobs.append((by_id[bench_id], pred))
    if args.limit is not None:
        jobs = jobs[:args.limit]
    if errors:
        print("Preflight failed:\n" + "\n".join(errors[:20]))
        return 2
    print(f"Preflight OK: {len(rows)} metadata rows, {sum(len(r['reference_slots']) for r in rows)} references, {len(jobs)} jobs")
    if args.preflight or args.dry_run:
        pending = sum(not result_valid(args.output_dir / p["generator"] / r["sample_id"] / RESULT) for r, p in jobs)
        print(f"Would evaluate {pending}; {len(jobs) - pending} valid results would resume")
        return 0
    client = APIClient(args.endpoint, args.api_key, args.model)
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(evaluate, row, pred, args.output_dir, client) for row, pred in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            failures += not result.get("success", False)
            if done % 25 == 0 or not result.get("success"):
                print(f"[{done}/{len(jobs)}] {result.get('bench_id')}: {'OK' if result.get('success') else result.get('error')}")
    print(f"Done: {len(jobs) - failures} succeeded, {failures} failed")
    return 1 if failures else 0
