"""Consume pending Image Generator rows in one-shot or follow mode."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..generator.runner import generate_row, iter_row_dirs, load_generator, row_complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-dir", type=Path, required=True)
    parser.add_argument("--generator-name", default="image")
    parser.add_argument("--backend", choices=("mock", "plugin"), default="plugin")
    parser.add_argument("--generator-plugin", default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--idle-exit-seconds", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generator = load_generator(args.backend, args.generator_plugin)
    failed = False
    idle_since = time.monotonic()
    while True:
        pending = [
            row
            for row in iter_row_dirs(args.lane_dir)
            if (row / "artifacts_files" / "s4_generation_manifest.json").is_file()
            and not row_complete(row, args.generator_name)
        ]
        if args.max_rows > 0:
            pending = pending[: args.max_rows]
        if pending:
            idle_since = time.monotonic()
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                results = executor.map(
                    lambda row: generate_row(
                        row,
                        generator_name=args.generator_name,
                        generator=generator,
                        model=args.model,
                    ),
                    pending,
                )
                for row, ok in zip(pending, results):
                    print(f"{row.name}: {'ok' if ok else 'failed'}")
                    failed = failed or not ok
        if not args.follow:
            return 1 if failed else 0
        if args.idle_exit_seconds > 0 and time.monotonic() - idle_since >= args.idle_exit_seconds:
            return 1 if failed else 0
        time.sleep(max(0.5, args.poll_interval))


if __name__ == "__main__":
    raise SystemExit(main())
