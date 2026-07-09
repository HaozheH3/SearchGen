# SearchGen Website — Design & Build Guide

A single-file, dependency-free project page for the paper *"Search Beyond What Can
Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation."* This doc
is enough to **reproduce the site from scratch, edit it, regenerate its figures, run
it locally, and deploy it.**

---

## 1. What the site is (architecture)

- **No build step, no framework, no dependencies.** Two hand-written, self-contained
  HTML files with inline `<style>` and `<script>`. Just static files served over HTTP.
- **`index.html`** — the academic paper page (hero → sections → citation).
- **`demo.html`** — the standalone animated "agentic trace" demo (Miyazaki visual
  search / Van Gogh no-search / CRISPR image+text search + stats outro). It is opened
  from `index.html` inside a modal `<iframe>`; it is never inlined.
- Only external network dependency: **Google Fonts** (Inter + JetBrains Mono). The
  page degrades gracefully to system fonts if fonts are blocked.

```
website/
├── index.html            # main paper page (self-contained)
├── demo.html             # animated demo (self-contained; opened in a modal iframe)
├── assets/
│   ├── figures/          # web-optimized PNGs converted from the paper's PDFs
│   │   ├── teaser.png  b1.png  search_hurt.png  e1.png  approach.png
│   │   ├── examples.png  treemap.png
│   ├── miya_1..5.*       # demo.html: Miyazaki reference images
│   ├── crispr_1..4.jpg   # demo.html: CRISPR reference images
│   └── demo1|demo2|demo3_none|oracle|result.png   # before/after outputs
├── WEBSITE_BUILD.md      # this file
└── (local-only, NOT pushed) CONTENT_DESIGN.md, WEBSITE_STRUCTURE_GUIDE.md
```

> **Content source of truth:** the section copy, captions, results table, and BibTeX
> come from `CONTENT_DESIGN.md` (kept local, not published — it carries internal
> review notes). `WEBSITE_STRUCTURE_GUIDE.md` is the reusable "how to lay out any
> paper site" playbook. If you edit copy, edit `index.html` directly; treat
> `CONTENT_DESIGN.md` as the rationale/reference.

---

## 2. Design system

Reproduce these exactly for visual consistency between `index.html` and `demo.html`.

**Type:** Inter (300–800) for text/UI; JetBrains Mono (400–600) for numbers, code,
BibTeX. Loaded via one Google Fonts `<link>`.

**Color tokens** (CSS custom properties on `:root`):

| Token | Value | Used for |
|---|---|---|
| `--accent` | `#0066ff` | brand blue (gradient start) |
| `--purple` | `#7c3aed` | brand violet (gradient end); eyebrows/kickers |
| `--text` / `--text2` / `--text3` | `#1a1a1a` / `#6b6b6b` / `#999` | body / secondary / tertiary |
| `--bg` / `--surface` / `--border` | `#fff` / `#f7f7f8` / `#e5e5ea` | page / cards / hairlines |
| `--green` (+`--green-lt` `#dcfce7`) | `#16a34a` | **ours / internalized / with-search** |
| `--orange` (+`--orange-lt`) | `#ea580c` | **naive / must-fetch / caution** |
| `--red` (+`--red-lt`) | `#dc2626` | **failure / deficit** |
| `--gold` (+`--gold-lt`) | `#ca8a04` | **no-search-needed (T2I)** |
| `--blue` (+`--blue-lt`) | `#2563eb` | text/web-search & info |

**Two hard rules** (carried from the design guide):
1. The **blue→purple gradient encodes structure only** (hero title, progress bar,
   launcher, section rules) — **never data**.
2. **Colorblind-safe:** hue is never the only channel. Findings/insight cards pair a
   colored left border with a label; the boundary diagram uses solid vs. hollow dots;
   charts keep numeric labels. Keep this when adding anything.

**Reusable components** (class names in `index.html`):
`.badge-link` / `.badge-link.alt` (hero links), `.badge-demo` / `.demo-fab` (demo
launchers), `section.card` (+`.narrow` for text-only sections), `.callout` findings
cards (green left-border) and the insight card (violet), `.results` table with the
emphasized Phase-2 row, `.stat-row`/`.sbox` (harness number tiles), `.eyebrow`
(section kicker), `pre code` (BibTeX).

---

## 3. Page structure of `index.html`

Order top→bottom (see `CONTENT_DESIGN.md` for the full copy of each):

0. `<div class="progress" id="progress">` — scroll-progress bar (fixed, top).
0. `<button class="demo-fab" id="demoFabTop">` — sticky "Live Demo" launcher (top-right).
0. `<header class="hero">` — kicker, `<h1>` title + `.sub` subtitle, `.authors`
   (superscript affiliations), `.tagline`, two `.badges` rows (arXiv / Project Page /
   GitHub, then the 🤗 dataset/corpus/bench), the mirrored `#demoFabHero` "Watch the
   Live Demo" button, and the disabled **"Play with the Interactive Demo (off-service
   now)"** entry.
0. `.hero-teaser` — static teaser figure (`assets/figures/teaser.png`).
Then `<main>` with these `section.card`s:
1. **TL;DR** (`.narrow`) — text only.
2. **The Hook** — before/after pairs (reuse `assets/demo1_none|oracle.png`,
   `demo3_none|oracle.png`) + the 12-category gallery (`assets/figures/examples.png`).
3. **Finding 1** — green callout card + `assets/figures/b1.png`.
4. **Finding 2** — green callout card + `assets/figures/search_hurt.png`.
5. **The knowledge boundary** — violet **Insight** card + the **inline SVG diagram**
   (`id="boundary"`, see §5).
6. **Method** — `assets/figures/approach.png`.
7. **Finding 3** — green callout card + `assets/figures/e1.png` + the **HTML results
   table** (Phase-2 `31.8` row bolded/green-tinted, "same generator · ⅟ reasoner
   cost" chip, naming note).
8. **The harness** — `.stat-row` tiles + `assets/figures/treemap.png`.
9. **Looking forward** (`.narrow`) — text.
10. **Citation** (`.narrow`) — BibTeX in `<pre><code id="bibtex">` + copy button.
`<footer>` — affiliations line.

---

## 4. The demo launcher (key interaction)

`demo.html` is opened from `index.html`, never inlined, so it always replays from the
start and never auto-plays on page load.

- **Triggers:** `#demoFabTop` (sticky pill) and `#demoFabHero` (hero button).
- **Modal markup:** `#demoModal` (fixed overlay) containing `#demoScrim` (dark
  backdrop), a frame wrapper, `#demoClose` (× button), and `<iframe id="demoFrame">`.
- **Open:** set `demoFrame.src = "demo.html"` (forces a fresh load → animation restarts),
  add `.open` to the modal, add `.modal-open` to `<body>` (locks scroll), focus `#demoClose`.
- **Close:** remove those classes and set `demoFrame.src = "about:blank"` (so it reloads
  next time). Closes on the × button, on `#demoScrim` click, and on **Esc**.
- **No-JS fallback:** a `<noscript>` link opens `demo.html` in a new tab.

---

## 5. The Knowledge-Boundary SVG (the one custom visual)

Section `id="boundary"`. Pure inline SVG + a little JS; no library.

- Layers: a light-gray disc ("all world knowledge"), a green disc ("internalizable"),
  a **dashed green circle** = the frontier (`#kbFrontier`, radius driven by JS),
  scattered dots in `#kbDots` (solid green inside the frontier / hollow orange
  outside), an outward `#kbArrow` labeled "+ co-training →", a legend, and a
  `#kbReplay` control.
- JS builds ~42 deterministic dots (seeded PRNG so layout is stable), then:
  - `setEndState()` — ships the **static expanded end-state by default** (frontier at
    `R1`, straddling dots flipped to green). This is what renders with no motion.
  - `kbAnimate()` — animates the frontier `R0→R1` and flips dots as it passes them;
    triggered once when the section scrolls into view, and by `#kbReplay`.
  - Gated by `prefers-reduced-motion`: reduced → static end-state only.

To retheme, change `R0`/`R1` and the green/orange fills; keep solid-vs-hollow dots.

---

## 6. Reveal-on-scroll (and why the page can't go blank)

Earlier a bug made the page appear empty: sections were `opacity:0` and only revealed
by JS, so if the observer didn't fire, nothing showed. **The rule now: content is
visible by default; the fade-in is a pure enhancement.**

- `section.card` is visible in base CSS. The hidden-then-fade state applies **only**
  under `html.js-anim` (a class JS adds at startup).
- JS uses an `IntersectionObserver` to add `.v` (fade in) as sections enter view, with:
  - a fallback to reveal everything if `IntersectionObserver` is unsupported, and
  - a **2.5s hard backstop** that reveals all sections no matter what.
- `@media (prefers-reduced-motion)` forces all sections visible.

If you add a new `section.card`, it inherits this automatically — do **not** set
`opacity:0` on it in base CSS.

---

## 7. Regenerating the figures

The `assets/figures/*.png` are rasterized from the paper's vector PDFs with
ImageMagick (delegates to Ghostscript). Requires `convert` + `gs`.

```bash
# from the paper's figures directory (source of the PDFs), output into website/assets/figures/
SRC=/path/to/paper/figures            # e.g. .../paper_materials/neurips_v7/figures
DST=/path/to/website/assets/figures
conv() { convert -density 200 -background white -alpha remove -alpha off -quality 92 "$SRC/$1" "$DST/$2"; }

conv searchgen_teaser.pdf teaser.png     # hero teaser + og:image
conv examples.pdf         examples.png   # §2 12-category gallery
conv b1_stratum_collapse.pdf b1.png      # §3 Finding 1 chart
conv search_hurt.pdf      search_hurt.png# §4 Finding 2 (copy-effect / concept-corruption)
conv e1.pdf               e1.png         # §7 progression (a) + boundary-shift CDF (b)
conv approach2.pdf        approach.png   # §6 method / co-training pipeline
conv treemap.pdf          treemap.png    # §8 domain long-tail
```

Notes:
- `-density 200` ≈ 2× for a ~900px content column. Raise to 300 for print-sharp; the
  large source figures (`examples`, `search_hurt`) were downscaled to ~2000–2200px
  wide to keep the page light.
- `examples.pdf`/`search_hurt.pdf` are photographic → keep PNG. Charts (`b1`, `e1`,
  `treemap`) *could* be rebuilt as inline SVG/Recharts from the source numbers if you
  want them on-palette (they currently keep the paper's original colors).
- The before/after and demo reference images under `assets/` are hand-curated (not
  generated by this script).

---

## 8. Run locally

```bash
cd website
python3 -m http.server 9898        # any free port
# open http://localhost:9898/
```

**VS Code Remote / dev-container caveat (important):** VS Code only auto-forwards
ports opened **from its own integrated terminal**. If you start the server from
another shell, the port won't tunnel to your browser and the page will spin forever.
Fix: either run the command **in the VS Code integrated terminal** (it auto-forwards),
or open the **PORTS** panel → *Forward a Port* → enter the port. The container binds
`0.0.0.0`, so both work once the port is forwarded.

Quick sanity check without a browser:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9898/          # expect 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9898/demo.html # expect 200
```

---

## 9. Deploy

Repo: **`git@github.com:HaozheH3/SearchGen.git`** (branch `main`). The site is static,
so GitHub Pages serves `index.html` from the repo root.

```bash
cd website
git add index.html demo.html assets/          # site files only
git commit -m "Update site"
GIT_SSH_COMMAND="ssh -i /path/to/id_ed25519 -o IdentitiesOnly=yes" git push origin main
```

In GitHub → Settings → Pages, set source to `main` / root. Public URL:
`https://haozheh3.github.io/SearchGen/`.

> Keep internal docs (`CONTENT_DESIGN.md`, `WEBSITE_STRUCTURE_GUIDE.md`) **out** of the
> commit — they contain review commentary not meant for the public repo.

---

## 10. Editing cheatsheet

| To change… | Do this in `index.html` |
|---|---|
| Title / authors / affiliations | `<header class="hero">` block |
| arXiv / Project / GitHub / HF links | the two `.badges` rows (`href` values) |
| A section's copy | that `section.card`'s `<p>` / `<h2>` |
| A finding/insight wording | the `.callout` card text |
| Results numbers | the HTML `<table>` in §7 (keep them matching the paper) |
| A figure | replace the file in `assets/figures/` (keep the name) or the `<img src>` |
| BibTeX | `<pre><code id="bibtex">` in §10 |
| Boundary diagram look | the `#boundary` SVG + its JS (`R0`,`R1`, fills) |
| The animated demo | `demo.html` (separate file; timeline in its `<script>`) |

**Metadata currently set:** arXiv `https://arxiv.org/abs/2607.05382`; GitHub
`https://github.com/HaozheH3/SearchGen`; HF dataset/corpus/bench links and Project
Page are placeholders (`href="#"`) — fill them when live. The interactive-playground
entry is intentionally disabled ("off-service now").
