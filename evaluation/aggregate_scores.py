#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from searchgen_eval.scoring import aggregate

p = argparse.ArgumentParser()
p.add_argument("output_dir", type=Path)
p.add_argument("--missing-policy", choices=["skip", "zero"], default="skip")
args = p.parse_args()
print(json.dumps(aggregate(args.output_dir, args.missing_policy), indent=2))
