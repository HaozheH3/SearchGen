# Evaluation protocol

The evaluation protocol reports checklist, adaptive rubric, prompt faithfulness, image quality, text rendering, AI naturalness, composition and aesthetics, physical plausibility, visual-reference evaluation, and text-reference evaluation. Scores use 0–3 with half points; non-applicable fields remain unscored according to the canonical parser.

The system prompt, prompt builder, PP insertion, interleaving, XML instructions, parser, and score extraction are vendored from the canonical implementation. Do not edit them as part of orchestration changes. Exact regression uses the golden fixture and stored raw responses; live responses are not expected to be deterministic.

Run preflight before API calls. Failed calls or parses produce a `success=false` diagnostic result. Writes are atomic. Resume occurs only when the parsed result exists, is valid JSON, and has `success=true`.
