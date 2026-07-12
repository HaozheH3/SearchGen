# Input and output schema

Public metadata uses `bench_id`, `user_prompt`, `verification_checklist`, `evaluation_rubric`, `visual_reference_slots`, and `text_knowledge_slots`. The loader maps `bench_id` to canonical `sample_id`, maps `visual_reference_slots` to `reference_slots`, and resolves each `relative_path` beneath `--benchmark-root`. Escaping paths and missing files are rejected. `release_row` is neither read nor required.

Prediction JSONL has one object per line:

```json
{"bench_id":"bench_0001","generator":"my_model","image_path":"/path/to/image.png"}
```

Relative image paths are resolved against the manifest directory. `--images-dir` is an alternative where each image basename is its `bench_id`.

The parsed result contains `success`, diagnostic `error`, `parsed`, flattened `scores`, `bench_id`, `sample_id`, generator name, and attached-reference count. Prompt and raw assistant response are saved verbatim beside it.
