"""Ensure ``artifacts_files/refined_prompt.txt`` exists (canonical top-level text for generators).

Resolution order for refined text (first hit wins):

1. ``result.json`` → ``pipeline_output.refined_prompt`` or top-level ``refined_prompt``
2. ``s3_refined_prompt.json`` → ``parsed_llm_output.refined_prompt`` (v2 protocol)
3. ``summary.json`` → ``pipeline_summary.refined_prompt`` (runs that omit ``result.json``)
4. Non-empty existing ``refined_prompt.txt``
5. ``user_prompt.txt`` (``none`` / identity lanes)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def resolve_refined_text(artifacts_dir: Path) -> str | None:
    rj = artifacts_dir / "result.json"
    if rj.is_file():
        try:
            data = json.loads(rj.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            po = data.get("pipeline_output")
            if isinstance(po, dict):
                t = po.get("refined_prompt")
                if isinstance(t, str) and t.strip():
                    return t.strip()
            t2 = data.get("refined_prompt")
            if isinstance(t2, str) and t2.strip():
                return t2.strip()
    s3 = artifacts_dir / "s3_refined_prompt.json"
    if s3.is_file():
        try:
            s3data = json.loads(s3.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            s3data = None
        if isinstance(s3data, dict):
            parsed = s3data.get("parsed_llm_output")
            if isinstance(parsed, dict):
                tr = parsed.get("refined_prompt")
                if isinstance(tr, str) and tr.strip():
                    return tr.strip()
            tr2 = s3data.get("refined_prompt")
            if isinstance(tr2, str) and tr2.strip():
                return tr2.strip()
            tr3 = s3data.get("refined_prompt_final")
            if isinstance(tr3, str) and tr3.strip():
                return tr3.strip()
    stop_m = artifacts_dir / "reasoner_stop_before_s2.json"
    if stop_m.is_file():
        try:
            md = json.loads(stop_m.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            md = None
        if isinstance(md, dict):
            u = md.get("user_prompt_raw")
            if isinstance(u, str) and u.strip():
                return u.strip()
    sj = artifacts_dir / "summary.json"
    if sj.is_file():
        try:
            sdata = json.loads(sj.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            sdata = None
        if isinstance(sdata, dict):
            ps = sdata.get("pipeline_summary")
            if isinstance(ps, dict):
                ts = ps.get("refined_prompt")
                if isinstance(ts, str) and ts.strip():
                    return ts.strip()
    cur = artifacts_dir / "refined_prompt.txt"
    if cur.is_file():
        t3 = cur.read_text(encoding="utf-8", errors="replace").strip()
        if t3:
            return t3
    up = artifacts_dir / "user_prompt.txt"
    if up.is_file():
        t4 = up.read_text(encoding="utf-8", errors="replace").strip()
        if t4:
            return t4
    return None


def sync_refined_prompt_txt(artifacts_dir: Path, dry_run: bool = False) -> bool:
    """Write ``refined_prompt.txt`` if resolvable. Returns True if file exists and non-empty after."""
    t = resolve_refined_text(artifacts_dir)
    if not t:
        return False
    dest = artifacts_dir / "refined_prompt.txt"
    if dry_run:
        return True
    if not dest.is_file() or dest.read_text(encoding="utf-8", errors="replace").strip() != t:
        dest.write_text(t + "\n", encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "artifacts_dirs",
        nargs="*",
        type=Path,
        help="``artifacts_files`` directories (default: stdin one path per line if empty)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--walk-output-root",
        type=Path,
        default=None,
        help="Walk an output tree and patch every artifacts_files directory under it",
    )
    args = ap.parse_args()

    dirs: list[Path] = list(args.artifacts_dirs)
    if args.walk_output_root:
        root = args.walk_output_root.resolve()
        dirs.extend(sorted(root.rglob("artifacts_files")))

    if not dirs:
        for line in sys.stdin:
            p = line.strip()
            if p:
                dirs.append(Path(p))

    n_ok = n_fail = 0
    for d in dirs:
        d = d.resolve()
        if not d.is_dir():
            n_fail += 1
            continue
        if sync_refined_prompt_txt(d, dry_run=args.dry_run):
            n_ok += 1
        else:
            n_fail += 1
    print(json.dumps({"patched_or_ok": n_ok, "failed": n_fail, "dry_run": args.dry_run}, indent=2))


if __name__ == "__main__":
    main()
