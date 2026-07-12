"""Run the Image Generator on one row or a complete materialized lane."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..generator.runner import generate_row, iter_row_dirs, load_generator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-dir", type=Path, required=True)
    parser.add_argument("--generator-name", default="image")
    parser.add_argument("--backend", choices=("mock", "plugin"), default="plugin")
    parser.add_argument("--generator-plugin", default=None, help="Trusted package.module:function")
    parser.add_argument("--model", default="")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--row-index", type=int)
    group.add_argument("--all-rows", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--allow-external-reference-paths", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generator = load_generator(args.backend, args.generator_plugin)
    if args.row_index is not None:
        rows = [args.lane_dir / f"eval_row_{args.row_index:03d}"]
    else:
        rows = iter_row_dirs(args.lane_dir)
    if not rows:
        raise SystemExit("no rows found")

    def run(row: Path) -> bool:
        return generate_row(
            row,
            generator_name=args.generator_name,
            generator=generator,
            model=args.model,
            width=args.width,
            height=args.height,
            allow_external_paths=args.allow_external_reference_paths,
            overwrite=args.overwrite,
        )

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                ok = bool(future.result())
            except Exception:
                ok = False
            print(f"{row.name}: {'ok' if ok else 'failed'}")
            if not ok:
                failures.append(row.name)
    print(f"complete={len(rows) - len(failures)} failed={len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
