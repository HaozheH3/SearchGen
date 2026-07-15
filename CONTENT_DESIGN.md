# SearchGen — Website Content Design (Draft)

> **What this document is.** A section-by-section draft of the SearchGen project
> website: the actual body copy paired with the exact visual that accompanies it,
> plus layout and text↔visual-balance notes. It reads like a storyboard for the
> page — an engineer (Lie) can build straight from it, and a writer/designer can
> edit copy and captions in place.
>
> **How it was made.** Following `WEBSITE_STRUCTURE_GUIDE.md`: we located the
> paper's five highlight boxes (Finding 1, Finding 2, the Insight, Finding 3, plus
> the worked evaluation example and the Knowledge-Boundary definition) and threaded
> them into one blog-style narrative — *problem → why the obvious fix fails → the
> key idea → method → evidence → what it enables* — then balanced each section with
> one carrying visual.
>
> **Source paper.** `Search Beyond What Can Be Taught: Evolving the Knowledge
> Boundary in Agentic Visual Generation` (neurips_v7). Every number is quoted
> from the paper.

---

## Page metadata

| Field | Value |
|---|---|
| **Title** | Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation |
| **Authors** | Haozhe Wang¹ · Weijia Feng³ · Jinpeng Yu³ · Che Liu⁴ · Ping Nie² · Fangzhen Lin¹ · Jiaming Liu³ ✉ · Ruihua Huang³ · Jimmy Lin² · Wenhu Chen² · Cong Wei² ✉ |
| **Affiliations** | ¹ Hong Kong University of Science and Technology · ² University of Waterloo · ³ Qwen Applications · ⁴ Imperial College London |
| **Corresponding** | Jiaming Liu, Cong Wei (✉) |
| **Badge links (hero)** | arXiv · Project Page · GitHub (`HaozheH3/SearchGen`) · 🤗 SearchGen-20K (dataset) · 🤗 SearchGen-Corpus-1M (search corpus) · 🤗 SearchGen-Bench |

**Canonical numbers (use these verbatim, site-wide):**
`20,839` prompts (SearchGen-20K, total) · `751`-prompt test set (SearchGen-Bench) ·
`20,188` training rows · `12` failure categories · `22` domains · `5.2` mean
knowledge gaps/prompt · `93.1%` of `31,537` entities appear in one prompt ·
open generators score `21–28 / 100` on search-intensive prompts (a ~`40`-point
collapse) · SearchGen-Corpus-1M = `145,642` search sessions, `559,973` unique URLs,
`370,733` cached downloads · `90,452` reasoning traces · `281,925` generations ·
co-trained Klein-4B `31.8` vs frontier oracle `31.2` · DPO `+2.8`, RFT `+2.6` ·
NoSearch selectivity `49.9 → 56.9` · full RFT cycle `4×8` GPU-hours.

---

## Section 0 · Hero — the animated agentic demo

**Layout:** full-bleed. **Balance: ~90% visual / 10% text.** Keep the existing
animated agentic-trace demo (`index.html`: Miyazaki = image search, Van Gogh =
skip, CRISPR = image+text) — it is the single strongest asset on the site and it
*shows* the thesis instead of stating it.

**Title (H1):**
> Search Beyond What Can Be Taught
> <span class="sub">Evolving the Knowledge Boundary in Agentic Visual Generation</span>

**Tagline (≤20 words):**
> **Image generators fabricate what they don't know. This one looks it up first — and knows when not to.**

**Lede (one short paragraph, above the animation):**
> Watch the agent think. In the demo below it reads three prompts and makes three
> different calls: search the web for a face it would otherwise invent, skip search
> entirely for a style it already knows cold, and pull both pictures *and* facts for
> a diagram it has to get exactly right. The interesting decision isn't *how* to
> search. It's *whether* to.

**Outro card line (keep the existing one):** *Searches when needed. Skips when not.*

**Alternate taglines (pick one):**
- When should an image model search the web — and when should it just draw?
- Generators render beautifully and hallucinate confidently. We taught one to tell the difference.
- Search beyond what can be taught.

**🖼 Visual**
- **Primary:** existing animated demo (reuse as-is).
- **Fallback / social preview:** `figures/searchgen_teaser.pdf` → **convert PNG@2×**, wired as `<noscript>` + `og:image`.
- **Fallback caption:** *Two paradigms for knowledge-hungry prompts. Left: prompt-rewriting inflates the text but still generates from stale weights. Right: SearchGen's agent fetches live web + visual context, then conditions the generator — grounding facts a model cannot know.*
- Keep the "Prompt Rewrite | SearchGen (Ours)" column labels so the split reads without color (colorblind-safe).

---

## Section 1 · TL;DR

**Layout:** narrow measure (≤640px), centered. **Balance: 100% text** (one inline
accent stat; no chart here — it would compete with Finding 1).

**H2:** The short version — teach what you can, search the rest

> Modern image generators render gorgeously and lie fluently. Ask for the 2025
> Osaka Expo mascot and you get a confident, wrong invention. The failure isn't the
> pixels — it's the *knowledge*. On **SearchGen-Bench**, frontier open generators
> score just **21–28 out of 100** on search-intensive prompts — up to a ~40-point
> collapse that standard benchmarks never register.
>
> Search is the obvious fix, the way an illustrator consults references. But naive
> search backfires: it corrupts prompts the generator already handled. The real
> problem is a **knowledge boundary** — the line between what a generator can learn
> and what it must look up. That line is generator-specific, it moves during
> training, and it can't be hand-drawn. It has to be *discovered*.
>
> So we discover it, by **co-training the generator and the search agent together**.
> Below: how the collapse works, why naive search fails, what the knowledge boundary
> is, and how a co-trained 8B reasoner on a 4B generator matches a frontier *reasoner*
> on the same generator — jump to the finding cards. ↓

---

## Section 2 · The Hook — generators fabricate what they don't know

**Layout:** two before/after pairs (half-width each; stack on mobile), gallery strip
below. **Balance: ~75% visual / 25% text.**

**H2:** They render beautifully. They just make things up.

> Ask a frontier image model for the mascot of the 2025 Osaka Expo. You get a
> polished, confident fabrication. Ask for a historically accurate Spartan phalanx
> and you get anachronistic armor, rendered in exquisite detail.
>
> The lighting is right. The composition is right. The *world* is wrong.
>
> This is not a rendering failure. It's a **world-knowledge bottleneck**. Generators
> train on fixed corpora with hard knowledge cutoffs; user requests draw on new
> characters, regional symbols, niche typography, historical artifacts, and events
> that postdate training.
>
> Worse, generators have no way to flag their own ignorance. They're trained to
> always output an image — never to say "I don't know what this looks like." So they
> guess, beautifully, every time.

**Card line (pull-quote):** *The lighting is right. The composition is right. The world is wrong.*

*Bridge to next section:* So the obvious move is to let the model look things up. First, let's measure exactly how far it falls without help. ↓

**🖼 Visual**
- **Primary (before/after):** `website/assets/demo1_none.png` vs `demo1_oracle.png` (Miyazaki), and `demo3_none.png` vs `demo3_oracle.png` (CRISPR). **Reuse** (already high-res).
- **Secondary (gallery strip):** `figures/examples.pdf` → **convert PNG@2×**, sliced into a horizontally-scrollable "12 failure categories" strip.
- **Caption:** *Ask for a specific person, a labeled scientific diagram, or live data and today's best generators confidently fabricate. Left of each pair: no search — a generic face, misspelled gene-editing labels. Right: the same generator, this time with search-augmented grounding.*
- **⚠ Build note:** the `*_oracle.png` "after" images are grounded by the frontier oracle reasoner. That is honest for this section (the point is *search helps*), but the caption must say "search-augmented" — not imply these are the co-trained model's outputs. If co-trained-model outputs exist, prefer them here.
- Keep red-border "Without Search" / green-border "With Search" **plus word labels** (never border color alone).

---

## Section 3 · Finding 1 — the collapse you can't see

**Layout:** full-width chart, generous whitespace. **Balance: ~60% visual / 40% text** (a "card": chart + a tight takeaway).

**H2:** A 40-point gap that no benchmark shows

> **Finding 1 — Generators that score comparably on standard prompts diverge by
> nearly 40 points when search-intensive world knowledge is required.**
>
> On prompts that need only what a model already learned, open and commercial
> generators land in the same band (67–75 out of 100). Turn to prompts that need
> live world knowledge, and the field splits: open generators crater to **21–28**,
> while commercial systems with built-in search barely move. Existing benchmarks
> test rendering inside known concepts, so they never see this gap at all.
>
> To surface it, we built **SearchGen-Bench**: 751 test prompts scored by a
> 9-component judge on a 0–100 scale, with separate dimensions for *knowledge* and
> for *rendering*. The split is the whole point. When Flux.2-Klein-9B scores 24.2 on
> knowledge checklists but stays high on image quality, the diagnosis is
> unambiguous — it can draw, it just doesn't know.

*Bridge to next section:* If the disease is missing knowledge, the cure is search. Except the cure has a side effect. ↓

**🖼 Visual**
- **Source:** `figures/b1_stratum_collapse.pdf` → **convert SVG + recolor** (PNG@2× acceptable if time-boxed). **[P0]**
- **Shows:** per-generator quality on NoSearch stratum vs Search-Intensive stratum across 9 generators; drops of −0.1 (GPT-Image-2) to −39.1 (Qwen-Image-2).
- **Caption:** *Generators score well on prompts they can answer from memory (gray). On prompts that require external knowledge (orange), every open generator collapses — up to a 40-point drop — while commercial systems with built-in search hold. The bottleneck is missing knowledge, not rendering skill.*
- **Recolor spec:** gray = NoSearch stratum · orange `#ea580c` = Search-Intensive stratum · −Δ labels in red `#dc2626`. (Drop the current blue — it collides with the brand accent. Gray/orange + numeric Δ is colorblind-safe.)

---

## Section 4 · The obvious fix — and why it backfires

**Layout:** full-width, 3-column trace. **Balance: ~65% visual / 35% text.**

**H2:** Search should help. Often it hurts.

> An illustrator handed an unfamiliar brief looks up references before drawing. Give
> the generator the same move — a reasoner spots knowledge gaps, search fills them,
> the results feed generation — and you have *agentic visual generation*. Natural.
> And, done naively, harmful.
>
> **Finding 2 — Naive search actively degrades prompts the generator already
> handles.**
>
> Search everything, blindly, and every generator gets *worse* on prompts that never
> needed help. Qwen-Image-2 drops from **70.7 to 60.4** on the no-search stratum — a
> **14.6% relative loss** on prompts it already aced.
>
> Two distinct failures explain it. **Concept corruption**: search fires on
> something the model already knew, and the retrieved reference overrides correct
> internal knowledge — a *gating* failure, searching when it shouldn't. **Copy
> effect**: a reference carries so much detail that the generator copies it wholesale
> instead of borrowing the one missing fact — a *filtering* failure, keeping too
> much.

**Card line:** *The cure and the disease depend entirely on the patient. Search helps only when the generator is actually missing something.*

*Bridge to next section:* So why does the same tool rescue one prompt and wreck another? Because of a line most pipelines never see. ↓

**🖼 Visual**
- **Source:** `figures/search_hurt.pdf` → **convert PNG@2×** (photographic; keep raster). **[P2]**
- **Shows:** blind retrieval leaking into the output — a reference boat copied verbatim; a search-result caption pasted as the artwork's text.
- **Caption:** *Search is not free. Fed raw, retrieved content leaks into the image — the model copies a reference boat verbatim, or pastes a search-result's caption as the artwork's text. Naive retrieval corrupts the very prompts a model could have answered alone.*
- Layout as 3 labeled columns: "Prompt → No-Search → Retrieved → Result." Badge each row with its failure mode — **copy effect** / **concept corruption** (orange). Thin orange outline over the copied element so the eye lands on the corruption.

---

## Section 5 · The key idea — the knowledge boundary  *(the hinge)*

**Layout:** full-width — one hero concept diagram (the CDF proof moves to §7, so the
hinge carries a single, unmissable visual). **Balance: ~55% visual / 45% text** (the
core definition lives here; text does real work).

**H2:** The line between what a model learns and what it must look up

> Some knowledge belongs *inside* the model. A character's canonical look, a flag's
> fixed geometry — stable, low-dimensional, learnable once and for all. Fire search
> for it and you only add noise.
>
> Other knowledge belongs *outside*. It changes faster than retraining cycles, sits
> too deep in the long tail to ever learn reliably, or needs per-request reasoning.
> For this, search isn't a nice-to-have — it's structurally necessary. And the tail
> is enormous: **93.1% of the 31,537 entities** in our data appear in just a single
> prompt. No feasible training set covers that.
>
> **Insight — Some knowledge is internalizable and search should not fire for it;
> other knowledge is contextual and search is structurally necessary.**
>
> We call the divide between those two sets the **knowledge boundary**. Here's the
> twist that organizes the entire paper: the boundary is *generator-specific*, and
> it *moves*. As a generator learns, concepts migrate from "must search" to "already
> knows." A search policy tuned for a weak generator is wrong for a strong one.
>
> So you can't hand-specify it. And you don't have to. The boundary is
> **discoverable** — it falls out of training the generator and the searcher
> together.

**Definition (styled pull-quote):** *For a given generator, world knowledge splits
in two: what it can absorb into its own weights (internalizable), and what must stay
in the prompt as retrieved context (contextual). That split is the knowledge
boundary — and it shifts every time the generator improves.*

**🖼 Visual A — Knowledge Boundary Diagram (NEW, build in SVG) [P0, highest leverage]**
- **Shows:** internalizable set (green, inside a dashed "frontier" circle) vs
  contextual set (orange, outside); co-training expands the frontier and flips
  straddling concepts from fetched→internalized.
- **Caption:** *A generator's knowledge has a boundary. Inside it (green) lives what
  the model can internalize — generate correctly with no search. Outside (orange)
  lies contextual knowledge that must be fetched. Co-training pushes the boundary
  outward, converting fetch-only prompts into internalizable ones; whatever remains
  outside is handled by the agent.*
- **Build spec (for Lie):** inline SVG + CSS, revealed on scroll via the existing
  `.v` IntersectionObserver.
  - Canvas ~640×440, responsive `viewBox`.
  - Layers: light-gray disc ("all world knowledge") → green-tint disc r₀≈38%
    ("internalizable") → **dashed green circle** at r₀ ("knowledge frontier") →
    outer orange-tint rim band ("always needs search: fresh facts, rare entities,
    live data") → ~40 dots: **solid green** inside, **hollow orange-stroked**
    outside (solid-vs-hollow = the colorblind-safe channel).
  - Animate once on reveal (+ "▸ replay"): frontier grows r₀→r₁ (≈38%→52%); ~5
    straddling dots crossfade hollow-orange → solid-green; an outward arrow labeled
    **"+ co-training →"** appears. Guard with `prefers-reduced-motion` (show the
    expanded end-state statically).
  - Legend: ● green = "internalized · no search" | ○ orange = "fetched · agentic
    search" | ┄ dashed = "knowledge boundary (moves with training)."
  - Tie-in annotation: *"the boundary moves outward — measured proof in the results ↓"*.
- **De-risk (Rams):** a concept diagram is hard to parse in one glance, so ship a
  **strong static end-state** (the expanded frontier, dots already flipped) as the
  default; the animation is an enhancement gated behind `prefers-reduced-motion`. The
  empirical CDF that *proves* the shift now lives in §7 next to the progression chart.

---

## Section 6 · Method — teach, then search

**Layout:** full-width diagram. **Balance: ~70% visual / 30% text** (a good method
diagram is near self-explanatory).

**H2:** Build a searcher that resists noise. Then move the boundary.

> The method has two halves. First, a searcher that doesn't poison the generator.
> Second, a training loop that finds — and expands — the boundary.
>
> **The noise-resistant reasoner: gate, filter, integrate.**
> - **Gate** — decide *whether* to search. Only gaps rated *critical* or *important*
>   trigger a query; the rest are dropped. At most three queries per prompt, or
>   *skip* entirely. This is the defense against concept corruption.
> - **Filter** — decide *what to keep*. Choose the reference that fills the specific
>   gap with the least extraneous baggage. This is the defense against the copy
>   effect.
> - **Integrate** — decide *how it enters*. Route visual references through language,
>   not raw pixels: "following Image 1, render the robe in teal and gold." The
>   generator borrows exactly what's named — and nothing else leaks in.
>
> **Teach-then-search co-training.** The boundary only moves when the generator's
> weights change, so we act on both sides, in order.
> - **Phase 0 — warm-start.** Supervised finetuning gives an 8B reasoner
>   (Qwen3-VL-8B) the gate–filter–integrate protocol. Competent, but
>   generator-agnostic: it searches for anything that *might* be hard.
> - **Phase 1 — teach the generator.** Online iterative Diffusion-DPO feeds it
>   search-augmented examples and reinforces its own best outputs. Two things happen
>   at once: it *internalizes* stable knowledge (the boundary pushes outward), and it
>   learns to use imperfect references without being dominated by them
>   (noise-robustness).
> - **Phase 2 — recalibrate the searcher.** The generator is now stronger, so the
>   search policy is stale. Rejection-sampling finetuning rewards trajectories where
>   search actually helped the *new* generator and discards the rest. Search fires
>   only for what remains genuinely contextual.

**Card line:** *Phase 1 moves the boundary outward; Phase 2 moves the search policy inward to match. The model is never told where the line is — it learns it from which searches actually helped.*

**🖼 Visual**
- **Source:** `figures/approach2.pdf` → **redraw for web (SVG)** — content is right but
  the paper figure is visually noisy (emoji icons, hand-drawn arrows, cramped
  legend). PNG@2× as fallback. **[P1]**
- **Caption:** *Two coupled loops. Bottom: an agent gates each prompt, fetches and
  filters references only when a knowledge gap warrants it, then integrates them into
  an enriched prompt. Top: online DPO teaches the generator what it can internalize —
  expanding the knowledge boundary so the agent has less to fetch over time.*
- **Redraw semantics:** upper "teach/internalize" loop = green; lower "search/fetch"
  loop = orange; DPO chosen/rejected pair marked with **✓/✗** (not color alone);
  dashed circle = "knowledge frontier" — **reuse the dashed style from Section 5** for
  visual rhyme.

---

## Section 7 · Finding 3 — the boundary is discoverable

**Layout:** chart (half/full-width) + HTML results table below.
**Balance: ~55% visual / 45% text** (the table row *is* the argument).

**H2:** Co-training discovers the boundary — an 8B reasoner matches a frontier oracle

> **Finding 3 — Co-training discovers the knowledge boundary: a calibrated 8B
> reasoner matches a frontier oracle on the same generator.**
>
> On Klein-4B, the full co-training progression climbs monotonically to **31.8
> overall** — matching the frontier Gemini oracle on the very same generator
> (**31.2**). Both phases pull their weight: teaching the generator adds **+2.8**,
> recalibrating the searcher adds **+2.6**.
>
> It's also *selective*, which is the hard part. On prompts that don't need search,
> the calibrated reasoner lifts the no-search baseline from **49.9 to 56.9** — it
> learned when to stay quiet, exactly where naive search did damage. And it's
> *generator-specific*: a policy tuned for the strengthened generator scores worse on
> the base one, confirming the boundary is a joint property of the pair, not a fixed
> property of the prompt.
>
> The headline is the calibration-per-dollar. An 8B reasoner on a 4B generator
> reaches the same quality as a frontier commercial reasoner on that generator — and
> the full recalibration cycle fits in **4×8 GPU-hours**.
>
> One honest caveat: this is a comparison of *reasoners on a fixed 4B generator*, not
> a claim of frontier image quality. In absolute terms, 31.8 still trails a
> search-integrated commercial system like GPT-Image-2 (**71.0**) by ~39 points —
> that gap is generator capacity at the 4B scale, and closing it means scaling the
> generator, not the recipe.

**Card line:** *Frontier-reasoner calibration on the same generator — at a fraction of the reasoner's size and training cost.*

**🖼 Visual A — progression chart**
- **Source:** `figures/e1.pdf` panel **(a)** → **SVG + recolor** (sequential green ramp: light→mid→dark for SFT→DPO→RFT). **[P0]**
- **Caption:** *Co-training compounds. Each round — reasoner SFT, generator DPO,
  reasoner RFT — lifts quality across all three difficulty tiers, for both Klein-4B
  and Bagel-7B.* (Keep value labels above every bar.)

**🖼 Visual B — the boundary shift, measured (moved here from §5)**
- **Source:** `figures/e1.pdf` panel **(b)** → extract right panel → **SVG + recolor**. **[P0]**
- **Caption:** *Proof the boundary moved: after co-training, the distribution of
  per-prompt no-search quality shifts right — more prompts now clear a given quality
  bar without any search. The shift holds for both Klein-4B and Bagel-7B.* (State the
  shift qualitatively; the paper reports a rightward CDF shift, not a headline
  delta — do not invent a mean.)
- Baseline = **solid gray** line, post-DPO = **dashed green** line (line style is the
  colorblind-safe channel).

**🖼 Visual C — main results (hand-built HTML table, never a screenshot)**
- **Source:** main results table (paper §4, ~line 827).
- **Emphasize the headline row**: bold `31.8` vs `31.2`, tint the row `--green-lt`, add a chip reading **"same generator · ⅟ reasoner cost"** (not "matched compute" — the reasoner compute is *not* matched; the generator is).
- **Caption:** *On a fixed Klein-4B-DPO generator, the co-trained 8B reasoner (31.8)
  matches the frontier oracle reasoner (31.2) — at a fraction of the reasoner cost.
  Training moved most of the knowledge inside the boundary.*

Minimal table to render (Overall column shown; expand with Easy/Medium/Hard + NoSearch). Note
the label: the SFT-8B reasoner is *generator-agnostic* — it gates like any reasoner but
is not calibrated to a specific generator. It is **not** the naive "search-everything"
policy from §4; avoid the word "blind" here to prevent that collision.

| | Config | n | NoSearch | Easy | Medium | Hard | Overall |
|---|---|---:|---:|---:|---:|---:|---:|
| Phase 0 | Gen-Agnostic (SFT-8B) + Klein-4B | 602 | 54.6 | 28.9 | 29.2 | 21.2 | 26.4 |
| Phase 1 | Gen-Agnostic (SFT-8B) + Klein-4B-DPO-v2 | 321 | 54.0 | 31.8 | 31.1 | 24.7 | 29.2 |
| **Phase 2** | **Gen-Adaptive (RFT-8B) + Klein-4B-DPO-v2** | **321** | **56.9** | **34.1** | **33.6** | **27.4** | **31.8** |
| ref | Oracle (frontier API) + Klein-4B-DPO-v2 | 750 | 55.7 | 33.7 | 33.9 | 26.0 | 31.2 |
| ref | No-Search + Klein-4B-DPO-v2 | 751 | 49.9 | 28.2 | 26.3 | 20.6 | 25.0 |

> Naming note (maps to the paper): the paper's condition macros render as **No Search
> → Blind Search (SFT-8B, generator-agnostic) → Generator-Adaptive Search
> (RFT-8B)**. We relabel "Blind Search" → "Gen-Agnostic" *on the site only*, because
> the paper reuses "blind" for the harmful search-everything policy in the Hook — two
> different meanings that must not collide on one page.

*Bridge to next section:* None of this is reproducible if you can't replay the searches. So we released them. ↓

---

## Section 8 · The harness — released, and replayable offline

**Layout:** treemap full-width; stat-tile row (4 tiles → 2×2 on mobile); table below.
**Balance: ~50% visual / 50% text.**

**H2:** Search-augmented generation you can reproduce without an API key

> Reproducing this kind of work usually means paying for a search engine *and* a
> fleet of generators — and watching your results drift as those services change
> underneath you. We froze the whole thing.
>
> We release **SearchGen-20K** (20,839 world-knowledge-grounded prompts across 12
> failure categories and 22 domains, a mean of 5.2 knowledge gaps per prompt), the
> co-training corpus (**90,452 reasoning traces**, **281,925 generations**), and
> **SearchGen-Corpus-1M** — **145,642 archived image and web search sessions**,
> **559,973 unique URLs**, and **370,733 cached downloads**.
>
> Because every search is pre-executed and frozen, you can replay the entire
> pipeline offline. No live API keys. No result drift. An expensive research workflow
> becomes a stable substrate for preference learning, reward modeling, search-policy
> design, and retrieval studies.

**Card line:** *A frozen web. Same query, same result, every run.*

**🖼 Visual A — stat-tile row (reuse existing `.sbox` component)**
> `20,839` prompts  ·  `145,642` search sessions  ·  `90,452` agentic traces  ·  `281,925` generations
- **Caption:** *Everything released: the prompts, the searches the agent ran, its full reasoning traces, and every generation used for training and evaluation.*

**🖼 Visual B — domain treemap**
- **Source:** `figures/treemap.pdf` → **SVG + recolor** (calm single-family qualitative palette; ensure label contrast ≥4.5:1). **[P1]**
- **Caption:** *SearchGen-20K spans 22 real-world domains — from People & Professions
  and Screen & Performance Media down a long tail through Science, Fashion, and
  Infrastructure — mirroring how people actually prompt.*

---

## Section 9 · Looking forward

**Layout:** half-width or inline. **Balance: ~80% text / 20% visual.** End light —
don't over-invest.

**H2:** A flywheel — and a principle bigger than search

> Our recipe is deliberately minimal: one teaching pass, one recalibration pass. Even
> so, it improves monotonically — which means it can *repeat*. Each cycle pushes the
> generator's boundary further out and tightens the search policy further in,
> converging toward a system where only genuinely contextual knowledge ever triggers
> a lookup. That's a recursive self-improvement flywheel for world-knowledge-grounded
> generation.
>
> The tempting objection is that bigger models will simply learn everything. They
> won't. Training data is finite; the world is not. No model, at any scale, can hold
> events after its cutoff, entities too rare for any dataset, or culture that keeps
> evolving. The boundary shifts outward with scale — it never disappears. Co-training
> finds where it lies for *any* generator, at *any* scale.
>
> And search is only the first tool. The same gate–filter–integrate discipline
> governs *when to invoke any tool* — image editing, render-as-code, 3D-asset
> retrieval, structural control. Each fills a different slice of what a generator
> can't be taught. The knowledge boundary is a general principle for tool use, and
> the released harness is built to explore it.

**Card line:** *The question isn't how to build a model that knows everything. It's how to build one that knows what it doesn't know.*

**🖼 Visual (optional, [P2]):** a compact "when to invoke a tool" decision-gate
schematic — `Prompt → [gap-severity meter] → below threshold → T2I (gold, "0
searches") | above → agentic search (green) | (ghost) → other tools (gray, dashed,
"future")`. Build from `.mchip` / `.action-callout` styles, or leave text-only.

---

## Section 10 · Citation

**H2:** Cite this work

```bibtex
@article{wang2026searchgen,
  title   = {Search Beyond What Can Be Taught: Evolving the Knowledge
             Boundary in Agentic Visual Generation},
  author  = {Wang, Haozhe and Feng, Weijia and Yu, Jinpeng and Liu, Che and
             Nie, Ping and Lin, Fangzhen and Liu, Jiaming and Huang, Ruihua and
             Lin, Jimmy and Chen, Wenhu and Wei, Cong},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://haozheh3.github.io/SearchGen}
}
```

**Footer:** SearchGen © 2026 · HKUST · University of Waterloo · Qwen Applications · Imperial College London

---

## Build backlog (rolled up from the visual plan)

| Priority | Asset | Action |
|---|---|---|
| **P0** | Knowledge Boundary Diagram (§5) | **Build new** (inline SVG + CSS) |
| **P0** | `b1_stratum_collapse` (§3) | Convert → SVG, recolor gray/orange + red Δ |
| **P0** | `e1` (a)+(b) (both in §7) | Convert → SVG, split into two charts, recolor |
| **P1** | `approach2` (§6) | Redraw clean for web (SVG) |
| **P1** | `examples` (§2) | Convert PNG@2×, slice into gallery scroller |
| **P1** | `treemap` (§8) | Convert → SVG, recolor for label contrast |
| **P2** | `search_hurt` (§4) | Convert PNG@2× + corruption-highlight overlay |
| **P2** | `searchgen_teaser` (§0) | Convert PNG@2× (static/OG fallback) |
| **P2** | decision-gate schematic (§9) | Optional build |

**Global visual system (unchanged from existing site — extend, don't replace):**
Inter + JetBrains Mono; brand gradient `#0066ff→#7c3aed` for **structure only, never
data**. Data-color semantics site-wide: **green = ours / internalized / with-help**,
**orange = naive / must-fetch / caution**, **red = failure / deficit**, **gold =
no-search-needed (T2I)**, **gray = baseline / NoSearch-stratum**, **blue =
text/web-search & info**. Colorblind rule: hue is *never* the only channel — always
pair with position, word labels, icons (✓/✗), line style (solid vs dashed), or direct
numeric labels.
