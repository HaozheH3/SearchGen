# The Paper-Website Structure Guide

*How to decide **what** goes on a paper's project website, and in **what order**.*

A reusable mindset + layout doc for turning a finished ML paper into a scrollable
project site. General enough for any paper; grounded in one worked reference (the
**RationalRewards** site, reverse-engineered at the end).

> **The medium.** A project website is not an A0 poster and not a PDF. It is a
> **vertical scroll**: one screen at a time, read at arm's length, by a visitor
> who decides in ~3 seconds whether to keep scrolling. Same discipline as a
> poster — one message, explicit hierarchy, whitespace as an active ingredient,
> color = meaning — but different physics. You do not fit everything on one
> surface; you **sequence** surfaces so each one earns the next scroll.

---

## 1. Mindset

Read these before you touch layout. Everything downstream is an application of
them.

### 1.1 A website is the paper *distilled into a scannable narrative*, not a compressed PDF
The paper argues, hedges, and proves. The website **announces and shows**. Your
job is not to shrink nine pages; it is to re-tell the paper as a visual-forward
story a stranger can follow while scrolling with one thumb. If your site reads
like the abstract with figures pasted in, you have built a worse PDF.

### 1.2 The one-message test
Before laying out anything, finish this sentence:
> *"If a visitor remembers exactly one thing from this site, it is ___."*

That sentence is the hero. It goes in the tagline, is proven by the teaser
figure, and is what every section either serves or gets cut. If you cannot write
it in one line, you are not ready to design — you are still reading.

### 1.3 Write like a blog post, not like an abstract
Abstracts front-load caveats and pack four ideas per sentence. Blog posts **hook
first, then deliver one point per paragraph**. On the web, lead each section with
the claim, then support it. Short paragraphs (2–4 sentences). No sentence should
require re-reading. Terse and confident beats complete and dense.

### 1.4 The three-read discipline (3s / 30s / 3min)
Design for three visitors at once — they are the same person at three depths:

| Read | Time | Who | What they must get | Carried by |
|------|------|-----|--------------------|------------|
| Scroll-past | ~3s | skims the whole page fast | the one message + that it's credible | title, tagline, hero figure, section headers |
| Skim | ~30s | reads headers + captions only | problem, key idea, headline result | H2/H3 headers, self-contained figcaptions, takeaway titles |
| Read | ~3min | genuinely interested | the argument + evidence | body prose, method figure, evidence cards |

**Test:** read *only* the headers and captions top-to-bottom. If that path alone
tells a coherent story, the site works. If it doesn't, the 30-second reader
leaves.

### 1.5 Visual–text balance: ~one visual per section, never a wall of text
Aim for roughly **one figure per content card**. A section that is all prose is a
section the skim-reader skips. A figure with no self-contained caption is a
figure the skim-reader can't parse. Prose sets up the point; the figure *is* the
evidence; the caption lets the figure stand alone. If a section has no natural
visual, it is probably two sections or half a section.

### 1.6 Forward-referencing pulls readers down the page
Each section should plant a reason to keep scrolling. The abstract card ends by
promising the insight cards below ("we highlight three findings in the cards that
follow"). The results teaser names a surprising number and implies the mechanism
comes next. Momentum is designed, not hoped for — every screen closes a small
loop and opens a slightly bigger one.

### 1.7 Glanceability and honest design
A figure read in one second must say something *true*. Cropped axes, cherry-picked
qualitative examples, and captions that overstate are the web equivalent of a
lie you'll get caught in during Q&A. Make the surprising result glanceable **and**
correct: if the headline needs an asterisk, put the asterisk in the caption, not
in a footnote no one scrolls to.

### 1.8 Earn the conversation
The site's job is to convert a scroller into a reader, a user, or a citer — not
to replace the paper. So surface the hooks (a bold number, a striking figure) and
the anchors that let someone act: arXiv, GitHub, released models/datasets, a demo,
the BibTeX. A paper with released assets and a live demo earns far more
conversations than one that ends in a wall. Put the assets where the excited
visitor is looking: in the hero, and again near the end.

### 1.9 Progressive disclosure
The site is the top layer. Proofs, full ablation grids, hyperparameters, and
notation live **one link deeper** (the paper, the appendix, the repo). Show the
result; link the derivation. Every "but what about…" should have a link, not a
paragraph. The page stays scannable because the depth is available, not present.

---

## 2. How to find the spine

A website needs a single narrative thread. Here is the repeatable recipe to
extract it from a finished paper.

### Step 0 — State the one message (§1.2)
Write the one-sentence "if they remember one thing" line. Everything hangs on it.

### Step 1 — Harvest the highlight boxes first
Papers already mark their own high points. Before reading linearly, **collect
every element the authors chose to emphasize**:

- the teaser figure (Figure 1) and its caption;
- the abstract's last two sentences (usually the contribution claim);
- bolded **findings / insights / takeaways / observations**;
- boxed definitions or a named method;
- the intro's contributions bullet list;
- table rows or numbers the text calls out as surprising ("notably…", "we find…").

These are the paper's own answer to "what matters." They will become your
tagline, your teaser, and your insight/finding cards almost verbatim (rewritten
blog-style). Start here and you inherit the authors' editorial judgment.

### Step 2 — Lay the highlights on the canonical thread
Almost every ML paper fits this logical chain. Slot each harvested highlight onto
it:

```
Problem  →  Why the obvious fix fails  →  Key idea  →  Method  →  Evidence  →  What it enables
```

- **Problem** — the pain, in one sentence. Why should anyone care today?
- **Why the obvious fix fails** — the tension that makes the paper non-trivial.
  (Often missing from papers but the strongest hook — surface it.)
- **Key idea** — the insight, promoted to its own "Why <core idea>?" section.
- **Method** — how the idea is realized; one figure.
- **Evidence** — the results that prove it; one figure per claim.
- **What it enables** — use cases, released assets, downstream wins.

If a highlight doesn't fit the thread, it's probably a "cut" (§6) or a caption
detail — not a section.

### Step 3 — Pick the single dominant element
One thing is the hero: usually the teaser figure or the single most surprising
result. It gets the most space, sits highest, and is what the tagline promises.
Everything else is second or third tier. If two things fight to be the hero, the
page has two messages — go back to Step 0.

### Step 4 — Order for momentum, then chain forward-references
Put the sections in the default order (§4), then write the last line of each
section to point at the next (§1.6). Read the header+caption path (§1.4) end to
end. Adjust until that path alone tells the story.

---

## 3. Canonical section inventory

Every candidate block a paper site might contain. Use it as a checklist: keep the
Must-haves, keep Optionals that serve *this* paper's message, cut the rest.

| # | Block | Purpose | Priority | Content source in the paper | Typical visual |
|---|-------|---------|----------|-----------------------------|----------------|
| 1 | **Hero: title + authors + affiliations** | Identity & credibility in 1s | Must | Title page | — (typographic) |
| 2 | **Tagline** (one sentence) | The one message, plain-language | Must | One-line contribution / last abstract sentence | — |
| 3 | **Badge / asset links** | Earn the conversation immediately | Must | arXiv, GitHub, HF models, HF datasets, demo | Row(s) of badges |
| 4 | **TL;DR / Abstract card** | The 30s version; forward-refs the rest | Must | Abstract, rewritten in 3–4 short paras | Optional small icon |
| 5 | **Teaser / "experience" block** | Make the one message glanceable | Must | Figure 1 + caption | Teaser figure or interactive |
| 6 | **Main result at a glance** | State the single surprising number up front | Must | Headline table/plot | Results teaser figure |
| 7 | **Hook: "Why the obvious fix fails"** | Create the tension the paper resolves | Optional (strong) | Intro motivation / related-work gap | Small diagram or none |
| 8 | **"Why <core idea>?" card** | Promote key insights to Takeaway 1/2/3 | Must (if paper has insights) | Findings / insight boxes | Usage / concept figure |
| 9 | **Method card** | How the idea works, in one glance | Must | Method section, distilled | One method/architecture figure |
| 10 | **Evidence cards** (×3–5) | One claim, one figure, each | Must | Experiments (one result each) | One plot/table per card |
| 11 | **Released assets / harness** | Let people use it | Optional (strong) | Artifacts, code, checkpoints | Screenshot / asset table |
| 12 | **Qualitative gallery** | Show, don't tell; extra use cases | Optional | Qualitative examples appendix | Grid of examples |
| 13 | **Limitations / future work** | Honesty; scope the claims | Optional | Limitations section | — |
| 14 | **Citation card** | Make citing frictionless | Must | BibTeX | Code block |
| 15 | **Footer** | Copyright, lab, contact | Must | — | — |

**Rules of thumb.** Blocks 1–6 are the "above/near the fold" spine that must land
the one message before a skeptic bails. Blocks 8–10 are the body of the argument.
Blocks 11–14 convert interest into action. If forced to cut, cut in reverse
priority: gallery → limitations → hook → evidence beyond three.

---

## 4. Ordering & flow

### 4.1 The default reading order

```
Hero (title · authors · tagline · badges)
  ↓  "here's what this is + it's real, links right here"
Abstract / TL;DR card  ──ends by promising the cards below──┐
  ↓                                                          │
Main results at a glance  (the surprising number, up front)  │ forward-ref
  ↓                                                          │
"Why <core idea>?"  →  Takeaway 1 / 2 / 3  ←─────────────────┘
  ↓
Method card  (one figure: how it works)
  ↓
Evidence card 1 … n  (one claim + one figure + one caption each)
  ↓
Released assets / qualitative gallery  ("what it enables")
  ↓
Citation  →  Footer
```

### 4.2 Why this order
- **Credibility before argument.** Authors, affiliations, and real links up top
  buy you the reader's attention for the rest.
- **Payoff before mechanism.** State the surprising result *before* the method
  (§ Main results). Web readers grant patience only after they see it's worth it.
  The paper can build up; the site cannot.
- **Idea before evidence.** The "Why <core idea>?" card frames what the evidence
  is evidence *of*, so each later plot lands as confirmation, not raw data.
- **Action last, where excitement peaks.** Assets, demo, and citation sit at the
  bottom because that's where a convinced reader is ready to act.

### 4.3 Where the one dominant element goes
The hero visual (teaser or headline result) sits in the **first or second screen**
— the 3-second reader must hit it without scrolling far. Give it more vertical
space and more whitespace than anything else on the page. Only one element gets
this treatment per page.

### 4.4 How forward-references chain the sections
Each section's closing line hands off to the next:
- Abstract → *"we highlight three findings in the cards below."*
- Main results → *"why does this happen? The key idea is X."*
- Key idea → *"here's how we realize it."* (→ Method)
- Method → *"does it work? The evidence:"* (→ Evidence cards)
- Last evidence card → *"what does this enable?"* (→ Assets / gallery)

The chain is the difference between a page someone scrolls and a page someone
*reads*.

---

## 5. Visual–text balance rules

1. **One visual per content card.** If a card has no visual, question whether it's
   a card. If it has three, split it.
2. **Every caption is self-contained.** A visitor who reads *only* the caption
   must understand what the figure shows and why it matters. Assume zero body-text
   context. Caption = claim + what's plotted + the takeaway. (This is the load-
   bearing rule for the 30-second reader.)
3. **Prose sets up; figure proves; caption stands alone.** Body text is the hook
   and one point; the figure is the evidence; the caption is the portable version.
   Don't repeat the caption in the body — say something else.
4. **Convert paper PDFs to web-native raster/vector.** Export figures to PNG at
   2× for retina (or SVG for diagrams and anything with text). Never embed a PDF
   or screenshot a PDF viewer. Re-crop for the screen: kill paper margins, bump
   font sizes, drop panel labels the web layout makes redundant.
5. **Redraw for the web when the paper figure won't glance.** Multi-panel figures
   dense enough for a 2-column PDF often fail at 3-second glance. Split them,
   enlarge the one panel that matters, or rebuild web-native.
6. **Build an interactive only when it beats a static image at the one message.**
   A demo, a hover-to-compare, or a slider is worth it when interaction *is* the
   evidence (e.g., "try the model," "scrub the training evolution"). If a static
   figure says it just as well, ship the static figure — interactives are cost.
7. **Color = meaning, consistently.** One accent that always means "ours / the
   point," neutrals for everything else, colorblind-safe, same palette in every
   figure. A consistent accent across figures does more than any single chart.
8. **Whitespace between cards is structure.** Generous vertical gaps tell the
   scroll-reader where one idea ends and the next begins. Crowding reads as panic.

---

## 6. What to cut

Paper content that should **not** appear on the site — and where it goes instead.

| Cut from the site | Why | Where it lives |
|-------------------|-----|----------------|
| Proofs & derivations | Nobody reads math while scrolling | Paper / appendix (link) |
| Most of related work | It's positioning, not the message | One-line framing in the hook; rest → paper |
| Hyperparameter tables | Reproducibility detail, not narrative | Appendix + repo README |
| Full ablation grids | Overwhelms the one message | Keep the one decisive ablation; rest → paper |
| Notation & preliminaries | Setup cost the web reader won't pay | Paper |
| Every baseline | Show the comparison that matters | Headline result; full table → paper |
| Exhaustive qualitative dumps | Diminishing returns | A curated gallery of the best few |
| Hedges & caveats | Blog voice is confident | A single honest limitations note (§13) |
| Dataset construction minutiae | Process, not payoff | Dataset card / repo |

**Principle:** the site shows *that* it's true and *why it matters*; the paper
shows *how* it's true. When tempted to add a paragraph of detail, add a link
instead (§1.9).

---

## 7. Pre-publish checklist

Run every item before shipping.

**The three-read test (§1.4)**
- [ ] **3s:** Open the top screen. Is the one message obvious from title +
      tagline + hero visual alone, with zero scrolling?
- [ ] **30s:** Read *only* headers + figcaptions top to bottom. Do they alone
      tell a coherent story (problem → idea → evidence → enables)?
- [ ] **3min:** Does the full read deliver the argument without a wall of text or
      a dead section?

**One message & flow**
- [ ] One dominant element per page; no two things fight to be the hero.
- [ ] The one-sentence message is stated in the tagline and proven by the teaser.
- [ ] Each section's last line forward-references the next.

**Visual–text balance**
- [ ] ~One visual per content card; no all-prose section.
- [ ] Every figcaption is self-contained (claim + what's shown + takeaway).
- [ ] All figures are web-native PNG@2× / SVG — no embedded or screenshotted PDFs.

**Glanceability & honesty (§1.7)**
- [ ] Every figure read in 1 second says something true (axes honest, no cherry-
      picking, caveats in the caption not hidden).
- [ ] No caption or headline overstates the paper.

**Earn the conversation (§1.8)**
- [ ] arXiv, GitHub, and all released models/datasets are linked in the hero.
- [ ] BibTeX is present and copy-pastable.
- [ ] **Every link works** — click each one, including HF and demo.

**Legibility & craft**
- [ ] Body line length 45–75 chars; comfortable line-height; readable on mobile.
- [ ] Consistent accent color = "ours/the point" across all figures; colorblind-safe.
- [ ] Generous whitespace between cards; nothing crowded.
- [ ] Renders on a phone (most first visits are mobile) and in dark mode if offered.

---

## Appendix A — Worked reference: the RationalRewards site

The reference site, reverse-engineered top→bottom, mapped to this guide.

| # | Section on the site | Guide block (§3) | Thread stage (§2) | Visual |
|---|---------------------|------------------|-------------------|--------|
| 1 | Hero: title, authors + superscript affiliations, one-sentence tagline, **two badge rows** (models row 1, datasets row 2: arXiv, project, GitHub, HF) | 1–3 | identity | typographic |
| 2 | Abstract card, 3–4 short paras; **last para forward-refs the insight cards below** | 4 | problem | — |
| 3 | Main Results at a Glance: short text + teaser figure; **most surprising result stated up front** | 6 | evidence (payoff first) | teaser figure |
| 4 | "Why <core idea>?" card: intro → **Takeaway 1 / 2 / 3** (why-question + answer) → usage figure | 8 | key idea | usage figure |
| 5 | Method card: short text + method figure + caption | 9 | method | method figure |
| 6–9 | Evidence cards: preference prediction, robustness, training evolution, test-time scaling — each 1–3 paras + **exactly one figure** + self-contained caption | 10 | evidence | one figure each |
| 10 | Additional use cases / qualitative gallery: text + figure | 12 | what it enables | gallery |
| 11 | Citation card: BibTeX code block | 14 | action | code block |
| 12 | Footer: copyright / lab | 15 | — | — |

**What to copy from it:**
- The **payoff-first** move — results at a glance *before* method.
- **Forward-referencing** from the abstract into the insight cards.
- Promoting the paper's **highlight boxes into Takeaway 1/2/3** (Step 1 → §8).
- **One figure per card**, every caption self-contained.
- **Two badge rows** separating released models from datasets — assets are
  first-class, surfaced in the hero (§1.8).
- Terse, blog-voice prose throughout — no paper sentences pasted in.

**The through-line:** RationalRewards is this guide's default order with the
"why the obvious fix fails" hook folded into the abstract and the key-idea card.
Any paper can be laid out the same way: harvest the highlights, lay them on
Problem → fix-fails → idea → method → evidence → enables, pick one hero, and chain
the sections with forward-references.
