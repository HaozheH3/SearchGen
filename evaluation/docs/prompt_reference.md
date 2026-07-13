# Prompt reference

SearchGen evaluation protocol version: `1.0`

The protocol always evaluates all ten components, including physical plausibility. The system prompt, user-prompt builder, multimodal image order, XML response instructions, parser, and score extraction are regression-tested together.

The fixture in `tests/fixtures/prompt_interleaved_golden.json.b64` contains a synthetic red-apple example. It freezes the complete system prompt, user prompt, and serialized interleaved request without including user data or production responses.

| Public artifact | SHA-256 |
|---|---|
| `searchgen_eval/judge_common.py` | `876f8df242f80c0386baab523f9ccf8592b03354423db739080697a83bcc9706` |
| `tests/fixtures/prompt_interleaved_golden.json.b64` | `02f19dcd0ed3be07305c81444005bd59c5cf1bbcc2b8c7c3c3ac8de491189ca3` |

Any intentional prompt or scoring change must increment the protocol version and update the golden fixture.
