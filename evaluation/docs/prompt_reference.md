# Prompt reference and provenance

Canonical sources frozen on 2026-07-12:

| Source | SHA-256 |
|---|---|
| `run_eval_648_release.py` | `889c6a7eee8e78a66f252674881f55477bc8f32e7c3bc4f3c6bd46df60702deb` |
| `toolgen_searchbetter_judge_common.py` | `29d7d2d997f424ac561378fedd5845c9b56793cd24398dd058f5fe367f44d63d` |
| `flow_factory/utils/image.py` | `fb6514a40ddda39b6986bebdeae6e50c2e58e763d90a5908b69374a78838a984` |
| `phase4_agent/frontier_model.py` | `79c10da56de18df3d213ee09e847ed58d97afbef5010a7cd61fc1023c49a87d0` |
| `compute_pp_score_tables.py` | `bc6f0d5569ccc794baa117fca69e4a1582775acbdff83efc1fde68d9b9344ebd` |

The judge source was vendored in full; only its relative image-helper import was changed. API and image operations were independently minimized. The golden fixture contains the complete system prompt, user prompt, and serialized interleaved request for deterministic fixed inputs. Template changes require a separately reviewed version change.
