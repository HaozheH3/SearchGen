<div align="center">

# Search Beyond What Can Be Taught

### Evolving the Knowledge Boundary in Agentic Visual Generation

Haozhe Wang<sup>1</sup> · Weijia Feng<sup>3</sup> · Jinpeng Yu<sup>3</sup> · Che Liu<sup>4</sup> · Ping Nie<sup>2</sup> · Fangzhen Lin<sup>1</sup> · Jiaming Liu<sup>3</sup>✉ · Ruihua Huang<sup>3</sup> · Jimmy Lin<sup>2</sup> · Wenhu Chen<sup>2</sup> · Cong Wei<sup>2</sup>✉

<sup>1</sup> HKUST · <sup>2</sup> University of Waterloo · <sup>3</sup> Qwen Applications · <sup>4</sup> Imperial College London

✉ Corresponding authors: Jiaming Liu, Cong Wei

[📄 arXiv](https://arxiv.org/abs/2607.05382) · [🌐 Project Page](https://haozheh3.github.io/SearchGen/) · [💻 GitHub](https://github.com/HaozheH3/SearchGen)

[🤗 SearchGen-20K](https://huggingface.co/datasets/JasperHaozhe/SearchGen-20K) · [🤗 SearchGen-Corpus-1M](https://huggingface.co/datasets/JasperHaozhe/SearchGen-Corpus-1M) · [🤗 SearchGen-Bench](https://huggingface.co/datasets/JasperHaozhe/SearchGen-Bench)

**Image generators fabricate what they don't know. _This one looks it up first—and knows when not to._**

<img src="https://haozheh3.github.io/SearchGen/assets/figures/teaser.png" width="900" alt="SearchGen teaser: prompt rewriting versus agentic retrieval">

*Two paradigms for knowledge-hungry prompts. **Left:** prompt rewriting inflates the text but still generates from stale weights. **Right:** SearchGen fetches live web and visual context, then conditions the generator—grounding facts a model cannot know.*

</div>

## TL;DR — Teach What You Can, Search the Rest

Modern image generators render gorgeously and lie fluently. Ask for the 2025 Osaka Expo mascot and you get a confident, wrong invention. The failure isn't the pixels—it's the **knowledge**. On **SearchGen-Bench**, frontier open generators score just **21–28 out of 100** on search-intensive prompts—up to a roughly 40-point collapse that standard benchmarks never register.

Search is the obvious fix, the way an illustrator consults references. But naive search backfires: it corrupts prompts the generator already handled. The real problem is a **knowledge boundary**—the line between what a generator can learn and what it must look up. That line is generator-specific, it moves during training, and it cannot be hand-drawn. It has to be discovered.

We discover it by **co-training the generator and the search agent together**. Below, we show how the collapse works, why naive search fails, what the knowledge boundary is, and how a co-trained 8B reasoner on a 4B generator matches a frontier reasoner on the same generator.

## They Render Beautifully. They Just Make Things Up.

Ask a frontier image model for the mascot of the 2025 Osaka Expo. You get a polished, confident fabrication. Ask for a historically accurate Spartan phalanx and you get anachronistic armor, rendered in exquisite detail.

The lighting is right. The composition is right. The **world is wrong**.

This is not a rendering failure. It is a **world-knowledge bottleneck**. Generators train on fixed corpora with hard knowledge cutoffs; user requests draw on new characters, regional symbols, niche typography, historical artifacts, and events that postdate training.

Worse, generators have no way to flag their own ignorance. They are trained to always output an image—never to say, “I don't know what this looks like.” So they guess, beautifully, every time.

<p align="center">
  <img src="https://haozheh3.github.io/SearchGen/assets/figures/examples.png" width="900" alt="Representative generation failures across SearchGen categories">
</p>

*Ask for a specific person, a labeled scientific diagram, or live data and today's best generators confidently fabricate. These representative cases span SearchGen's 12 failure categories.*

## Finding 1 — A 40-Point Gap That No Benchmark Shows

> **Generators that score comparably on standard prompts diverge by nearly 40 points when search-intensive world knowledge is required.**

On prompts that need only what a model already learned, open and commercial generators land in the same band (**67–75 out of 100**). Turn to prompts that need live world knowledge, and the field splits: open generators crater to **21–28**, while commercial systems with built-in search barely move. Existing benchmarks test rendering inside known concepts, so they never see this gap at all.

To surface it, we built **SearchGen-Bench**: 751 test prompts scored with separate dimensions for **knowledge** and **rendering**. The split is the whole point. When a generator scores poorly on knowledge checklists but remains strong on image quality, the diagnosis is unambiguous—it can draw; it just does not know.

<p align="center">
  <img src="https://haozheh3.github.io/SearchGen/assets/figures/b1.png" width="900" alt="SearchGen-Bench results across no-search and search-intensive prompts">
</p>

***SearchGen-Bench results.** Generators score well on prompts they can answer from memory (gray). On prompts requiring external knowledge (orange), every open generator collapses—up to a 40-point drop—while commercial systems with built-in search hold. The bottleneck is missing knowledge, not rendering skill.*

### Full Benchmark Breakdown

Scores are reported on a 0–100 scale; higher is better. **Checklist**, **Rubric**, and **Visual ref.** are knowledge-sensitive measures that test whether requested facts are present. **Image quality**, **Text rendering**, and **Physical plausibility** capture rendering competence. Together, the components distinguish a model that cannot draw from one that can draw but lacks the required world knowledge.

| Stratum | Type | Generator | Overall | Checklist | Rubric | Prompt | Image quality | Text rendering | AI naturalness | Composition | Physical plausibility | Visual ref. |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NoSearch | Open | Bagel | **49.3** | 52.3 | 49.1 | 39.2 | 58.0 | 18.4 | 48.3 | 65.3 | 60.4 | 34.8 |
| NoSearch | Open | Flux.2-Klein-4B | **51.0** | 54.7 | 51.8 | 39.3 | 61.3 | 13.2 | 49.5 | 67.2 | 63.5 | 32.6 |
| NoSearch | Open | Flux.2-Klein-9B | **57.8** | 63.8 | 59.8 | 51.3 | 63.8 | 31.3 | 52.5 | 72.3 | 66.7 | 43.7 |
| NoSearch | Open | Qwen-Image | **67.4** | 74.8 | 70.8 | 62.8 | 68.3 | 63.0 | 61.0 | 76.8 | 73.8 | 56.7 |
| NoSearch | Commercial | Qwen-Image-2 | **70.7** | 78.7 | 73.8 | 68.4 | 70.2 | 71.7 | 60.8 | 80.3 | 75.6 | 60.0 |
| NoSearch | Commercial | SeedDream-4.0 | **67.9** | 74.9 | 70.6 | 61.7 | 69.5 | 71.1 | 59.7 | 80.3 | 73.9 | 56.7 |
| NoSearch | Commercial | Nano Banana | **63.1** | 71.1 | 67.5 | 59.7 | 66.3 | 42.8 | 56.5 | 76.3 | 68.5 | 53.8 |
| NoSearch | Commercial | Nano Banana Pro | **75.0** | 82.8 | 78.1 | 72.8 | 71.5 | 85.9 | 65.0 | 83.3 | 78.2 | 67.8 |
| NoSearch | Commercial | GPT-Image-2 | **71.1** | 78.6 | 75.6 | 70.8 | 69.0 | 67.7 | 60.8 | 77.7 | 73.9 | 64.3 |
| Search-Intensive | Open | Bagel | **21.5** | 18.2 | 17.6 | 13.3 | 30.5 | 2.5 | 29.3 | 33.6 | 36.8 | 13.5 |
| Search-Intensive | Open | Flux.2-Klein-4B | **24.1** | 19.8 | 18.4 | 12.4 | 37.2 | 4.2 | 33.6 | 39.3 | 46.2 | 11.9 |
| Search-Intensive | Open | Flux.2-Klein-9B | **26.7** | 24.2 | 23.1 | 17.2 | 36.8 | 7.2 | 32.9 | 40.4 | 48.6 | 16.9 |
| Search-Intensive | Open | Qwen-Image | **27.9** | 24.8 | 24.3 | 18.6 | 40.1 | 8.7 | 31.6 | 42.8 | 44.6 | 17.7 |
| Search-Intensive | Commercial | Qwen-Image-2 | **31.6** | 28.5 | 27.1 | 22.1 | 42.2 | 12.7 | 36.3 | 45.5 | 48.2 | 21.0 |
| Search-Intensive | Commercial | Imagen3-Fast | **14.1** | 9.7 | 10.0 | 6.9 | 22.2 | 1.4 | 21.4 | 23.2 | 23.7 | 7.0 |
| Search-Intensive | Commercial | SeedDream-4.0 | **45.9** | 44.2 | 43.6 | 38.5 | 57.0 | 35.9 | 47.1 | 58.7 | 64.0 | 35.1 |
| Search-Intensive | Commercial | Nano Banana | **44.1** | 41.0 | 40.4 | 36.0 | 57.1 | 28.0 | 47.4 | 61.5 | 65.5 | 33.2 |
| Search-Intensive | Commercial | Nano Banana Pro | **65.3** | 64.4 | 63.1 | 60.7 | 71.4 | 65.0 | 62.0 | 75.9 | 78.5 | 58.3 |
| Search-Intensive | Commercial | GPT-Image-2 | **71.0** | 71.2 | 70.1 | 69.2 | 75.1 | 75.9 | 64.7 | 80.4 | 77.3 | 66.0 |

*NoSearch contains prompts for which parametric knowledge is sufficient; Search-Intensive contains prompts that require external knowledge. Imagen3-Fast was reported only on the Search-Intensive stratum.*

## Finding 2 — Search Should Help. Often It Hurts.

An illustrator handed an unfamiliar brief looks up references before drawing. Give the generator the same move—a reasoner spots knowledge gaps, search fills them, and the results feed generation—and you have **agentic visual generation**. Natural. And, done naively, harmful.

> **Naive search actively degrades prompts the generator already handles.**

Search everything blindly, and every generator gets worse on prompts that never needed help. Qwen-Image-2 drops from **70.7 to 60.4** on the no-search stratum—a **14.6% relative loss** on prompts it already aced.

Two distinct failures explain it. **Concept corruption:** search fires on something the model already knew, and the retrieved reference overrides correct internal knowledge—a gating failure. **Copy effect:** a reference contains so much detail that the generator copies it wholesale instead of borrowing the one missing fact—a filtering failure.

<p align="center">
  <img src="https://haozheh3.github.io/SearchGen/assets/figures/search_hurt.png" width="900" alt="Examples where naive search harms image generation">
</p>

***Search is not free.** Fed raw, retrieved content leaks into the image—the model copies a reference boat verbatim or pastes a search-result caption into the artwork. Naive retrieval corrupts the very prompts a model could have answered alone.*

## The Knowledge Boundary

Some knowledge belongs **inside** the model. A character's canonical look or a flag's fixed geometry is stable, low-dimensional, and learnable once and for all. Fire search for it and you only add noise.

Other knowledge belongs **outside**. It changes faster than retraining cycles, sits too deep in the long tail to learn reliably, or needs per-request reasoning. For this, search is structurally necessary. The tail is enormous: **93.1% of the 31,537 entities** in our data appear in just one prompt. No feasible training set covers that.

> **Some knowledge is internalizable and search should not fire for it; other knowledge is contextual and search is structurally necessary.**

We call the divide between those sets the **knowledge boundary**. It is generator-specific, and it moves. As a generator learns, concepts migrate from “must search” to “already knows.” A search policy tuned for a weak generator is wrong for a strong one.

The boundary cannot be hand-specified. It is **discoverable**: it falls out of training the generator and searcher together.

## Method — Gate, Filter, Integrate

The method has two halves: a searcher that does not poison the generator, and a training loop that finds and expands the boundary.

- **Gate:** decide *whether* to search. Only critical or important gaps trigger a query; the rest are dropped. Use at most three queries per prompt, or skip search entirely.
- **Filter:** decide *what to keep*. Choose the reference that fills the specific gap with the least extraneous baggage.
- **Integrate:** decide *how it enters*. Route visual references through language, not raw pixels, so the generator borrows exactly what is named and nothing else leaks in.

The **teach-then-search co-training** recipe acts on both sides of the boundary:

1. **Warm-start:** supervised fine-tuning gives an 8B reasoner the gate–filter–integrate protocol.
2. **Teach the generator:** online iterative Diffusion-DPO helps it internalize stable knowledge and use imperfect references robustly.
3. **Recalibrate the searcher:** rejection-sampling fine-tuning rewards trajectories where search helps the improved generator and discards the rest.

Phase 1 moves the boundary outward; Phase 2 moves the search policy inward to match. The model is never told where the line is—it learns from which searches actually helped.

<p align="center">
  <img src="https://haozheh3.github.io/SearchGen/assets/figures/approach.png" width="900" alt="SearchGen teach-then-search co-training approach">
</p>

***Two coupled loops.** The agent gates each prompt, fetches and filters references only when a knowledge gap warrants it, then integrates them into an enriched prompt. Online DPO teaches the generator what it can internalize, expanding the knowledge boundary so the agent has less to fetch over time.*

## Finding 3 — The Boundary Is Discoverable

> **Co-training discovers the knowledge boundary: a calibrated 8B reasoner matches a frontier oracle on the same generator.**

On Klein-4B, the full co-training progression climbs monotonically to **31.8 overall**, matching the frontier Gemini oracle on the same generator (**31.2**). Both phases pull their weight: teaching the generator adds **+2.8**, and recalibrating the searcher adds **+2.6**.

It is also selective, which is the hard part. On prompts that do not need search, the calibrated reasoner lifts the no-search baseline from **49.9 to 56.9**—it learned when to stay quiet, exactly where naive search did damage. It is also generator-specific: a policy tuned for the strengthened generator scores worse on the base one, confirming that the boundary belongs to the generator–reasoner pair, not to the prompt alone.

The headline is calibration per dollar. An 8B reasoner on a 4B generator reaches the same quality as a frontier commercial reasoner on that generator, and the full recalibration cycle fits in **4 × 8 GPU-hours**.

<p align="center">
  <img src="https://haozheh3.github.io/SearchGen/assets/figures/e1.png" width="900" alt="SearchGen co-training progression and knowledge boundary shift">
</p>

***(a) Co-training compounds.** Each round—reasoner SFT, generator DPO, reasoner RFT—lifts quality across all three difficulty tiers for Klein-4B and Bagel-7B. **(b) Proof the boundary moved:** after co-training, per-prompt no-search quality shifts right; more prompts clear a given quality bar without search.*

This compares reasoners on a fixed 4B generator; it is not a claim of frontier absolute image quality. In absolute terms, 31.8 remains below a search-integrated commercial system such as GPT-Image-2 (71.0). Closing that gap requires scaling the generator, not changing the recipe alone.

## The Released Harness

Reproducing search-augmented generation usually means paying for a search engine and a fleet of generators—and watching results drift as those services change. We froze the whole thing.

We release **SearchGen-20K** (20,839 world-knowledge-grounded prompts across 12 failure categories and 22 domains, with 5.2 knowledge gaps per prompt on average), the co-training corpus (**90,452 reasoning traces** and **281,925 generations**), and **SearchGen-Corpus-1M** (**145,642 archived image and web search sessions**, **559,973 unique URLs**, and **370,733 cached downloads**).

Because every search is pre-executed and frozen, the pipeline can be replayed offline. No live search API keys. No result drift. An expensive research workflow becomes a stable substrate for preference learning, reward modeling, search-policy design, and retrieval studies.

<p align="center">
  <img src="https://haozheh3.github.io/SearchGen/assets/figures/treemap.png" width="900" alt="The 22 domains in SearchGen-20K">
</p>

***SearchGen-20K spans 22 real-world domains**—from People & Professions and Screen & Performance Media down a long tail through Science, Fashion, and Infrastructure—mirroring how people actually prompt.*

## Repository Layout

```text
./
├── README.md
├── LICENSE
├── agent/
│   ├── README.md                 # API and input/output contracts
│   ├── QUICKSTART.md             # offline end-to-end demonstration
│   ├── pyproject.toml
│   ├── src/searchgen_agent/      # Agentic Reasoner and Image Generator
│   ├── examples/                 # custom API adapter templates
│   └── tests/                    # offline integration tests
└── evaluation/
    ├── evaluate.py                 # evaluation entry point
    ├── aggregate_scores.py         # result aggregation
    ├── searchgen_eval/             # evaluator implementation
    ├── docs/                       # protocol, schemas, and prompt reference
    ├── examples/                   # manifest and environment templates
    └── tests/                      # regression and end-to-end tests
```

## Agentic Reasoner with Search Tools and Image Generator

The [`agent/`](agent/) package contains:

- an **Agentic Reasoner with Search Tools** that analyzes requests, conditionally performs web/image search, selects useful references, and writes a refined generation manifest;
- an **Image Generator** that consumes the manifest through a user-defined image API plugin; and
- user extension points for custom chat, search, authentication, and image-generation services.

The agent package defines the complete adaptation contracts:

- [input dataset and reasoner output protocol](agent/README.md#1-agentic-reasoner-input-protocol);
- [OpenAI-compatible chat-model protocol](agent/README.md#2-chat-model-api-protocol);
- [custom web/image search request and response protocol](agent/README.md#3-custom-search-api);
- [`s4_generation_manifest.json` handoff protocol](agent/README.md#4-agentic-reasoner-output-protocol);
- [custom Image Generator input and output protocol](agent/README.md#5-custom-image-generator-api); and
- [generated image and metadata output protocol](agent/README.md#6-image-generator-output-protocol).

Install it with Python 3.10 or newer:

```bash
cd agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Follow [`agent/QUICKSTART.md`](agent/QUICKSTART.md) for a fully offline run. To connect real services, implement the search signer and image-generator callback shown in [`agent/examples/`](agent/examples/).

## Evaluation Environment

The evaluator requires Python 3.10 or newer:

```bash
cd evaluation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp examples/env.example .env
```

Set an OpenAI-compatible endpoint and judge model credentials in `.env`:

```bash
SEARCHGEN_EVAL_API_URL=https://your-openai-compatible-host.example/v1
SEARCHGEN_EVAL_API_KEY=replace_me
```

Then export them with `set -a; source .env; set +a`.

## Evaluation Quick Start

Download [SearchGen-Bench](https://huggingface.co/datasets/JasperHaozhe/SearchGen-Bench), then prepare a predictions manifest following [`evaluation/examples/generated_images_manifest.example.jsonl`](evaluation/examples/generated_images_manifest.example.jsonl). From the repository's `evaluation/` directory, validate all inputs without API calls, replacing the benchmark paths with the location of your download:

```bash
python evaluate.py \
  --metadata /path/to/SearchGen-Bench/eval_metadata.jsonl \
  --benchmark-root /path/to/SearchGen-Bench \
  --predictions-manifest predictions.jsonl \
  --output-dir results \
  --model your-judge-model \
  --preflight
```

Remove `--preflight` and add `--workers 16` to run evaluation. Completed successful examples resume automatically. Use `--dry-run` to inspect pending jobs, or repeat `--bench-id` and `--generator` to select subsets.

## Evaluation

SearchGen-Bench separates no-search and search-intensive prompts so evaluation can distinguish missing world knowledge from rendering failures and damage caused by unnecessary retrieval.

The released evaluation protocol reports checklist satisfaction, adaptive-rubric satisfaction, prompt faithfulness, image quality, text rendering, AI naturalness, composition and aesthetics, physical plausibility, visual-reference consistency, and text-reference consistency. Component scores use a 0–3 scale with half points; non-applicable fields remain unscored.

Aggregate completed results with:

```bash
python aggregate_scores.py results --missing-policy skip
```

See the **[`evaluation/` folder](evaluation/)** and its **[`README.md`](evaluation/README.md)** for the complete commands, input/output schema, evaluation protocol, prompt reference, and tests.

## License

The code release is licensed under the [Apache License 2.0](LICENSE).

## Citation

```bibtex
@article{wang2026searchgen,
  title   = {Search Beyond What Can Be Taught: Evolving the Knowledge
             Boundary in Agentic Visual Generation},
  author  = {Wang, Haozhe and Feng, Weijia and Yu, Jinpeng and Liu, Che and
             Nie, Ping and Lin, Fangzhen and Liu, Jiaming and Huang, Ruihua and
             Lin, Jimmy and Chen, Wenhu and Wei, Cong},
  journal = {arXiv preprint arXiv:2607.05382},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.05382}
}
```
