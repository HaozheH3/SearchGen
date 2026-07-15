# Demo GIF Capture Guide

This guide documents how `assets/searchgen_demo.gif` was created from the real
animated `demo.html` page. The page was rendered in headless Chromium, recorded
as WebM, and converted to a compact looping GIF with FFmpeg.

## Result

- Source: `demo.html`
- Output: `assets/searchgen_demo.gif`
- Dimensions: 900 × 570 px
- Duration: 14.3 seconds
- Frame rate: 12 fps
- Size: approximately 5.8 MB
- Content: hero and the first visual-search workflow
- Playback: approximately 3× faster than the live page

The full demo timeline is about 85 seconds. Capturing it at normal speed would
produce a very large GIF, so the published preview uses only the opening sequence
and accelerates it during encoding.

## Requirements

- Python 3, to serve the static site
- Node.js and npm
- Playwright with Chromium
- FFmpeg and FFprobe

Install Playwright and its Chromium runtime:

```bash
npm install --prefix /tmp/searchgen-gif-tools playwright@1.55.0
/tmp/searchgen-gif-tools/node_modules/.bin/playwright install chromium
```

On a minimal Linux machine, Chromium may also require system libraries:

```bash
/tmp/searchgen-gif-tools/node_modules/.bin/playwright install-deps chromium
```

The final command installs operating-system packages and may require root access.

## 1. Serve the website

Run the server from the `website` directory so relative asset paths resolve:

```bash
cd /path/to/finalized_release/website
python3 -m http.server 9898 --bind 127.0.0.1
```

Verify both pages are available:

```bash
curl -I http://127.0.0.1:9898/
curl -I http://127.0.0.1:9898/demo.html
```

## 2. Record the rendered page

Create `/tmp/capture_demo.cjs`:

```javascript
const { chromium } = require(
  '/tmp/searchgen-gif-tools/node_modules/playwright'
);

(async () => {
  const captureDir = '/tmp/searchgen-capture';
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1200, height: 760 },
    deviceScaleFactor: 1,
    recordVideo: {
      dir: captureDir,
      size: { width: 1200, height: 760 },
    },
  });

  const page = await context.newPage();
  const video = page.video();

  await page.goto('http://127.0.0.1:9898/demo.html', {
    waitUntil: 'networkidle',
  });

  // Long enough to include the first prompt, reasoning, image search,
  // reference selection, and visual-reference integration.
  await page.waitForTimeout(30000);

  await page.close();
  await context.close();
  console.log(await video.path());
  await browser.close();
})();
```

Record the page:

```bash
rm -rf /tmp/searchgen-capture
mkdir -p /tmp/searchgen-capture
node /tmp/capture_demo.cjs
```

Playwright writes a `.webm` file into `/tmp/searchgen-capture/`. Video recording
starts when the browser context is created, so it can include a short blank period
while the page, fonts, and images load. The encoding command below trims that
period.

Inspect the raw recording:

```bash
ffprobe -v error \
  -show_entries format=duration,size \
  -of default=nw=1 \
  /tmp/searchgen-capture/*.webm
```

## 3. Convert WebM to GIF

From the release root, run:

```bash
ffmpeg -y -i /tmp/searchgen-capture/*.webm \
  -filter_complex \
  "[0:v]trim=start=2,setpts=(PTS-STARTPTS)/3,fps=12,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
  -loop 0 \
  website/assets/searchgen_demo.gif
```

What the filter does:

- `trim=start=2` removes the initial two seconds of loading/blank video.
- `setpts=(PTS-STARTPTS)/3` accelerates playback by 3×.
- `fps=12` limits the frame count and file size while retaining smooth motion.
- `scale=900:-1` produces a 900 px-wide GIF and preserves the aspect ratio.
- `palettegen=max_colors=192` creates a palette tuned to the captured frames.
- `paletteuse` applies that palette with dithering and updates only changed areas.
- `-loop 0` makes the GIF loop indefinitely.

Using FFmpeg's two-pass palette filters is important. Direct conversion to GIF
usually produces worse colors, banding, and a larger file.

## 4. Verify the output

```bash
ls -lh website/assets/searchgen_demo.gif

ffprobe -v error \
  -show_entries format=duration,size \
  -of default=nw=1 \
  website/assets/searchgen_demo.gif
```

An optional contact sheet makes it easy to check the sequence in a headless
environment:

```bash
ffmpeg -y -v error \
  -i website/assets/searchgen_demo.gif \
  -vf "fps=1/3,scale=450:-1,tile=3x2" \
  -frames:v 1 \
  /tmp/searchgen-gif-check.png
```

## Tuning

To make the GIF smaller:

- Reduce `scale=900:-1` to `scale=720:-1`.
- Reduce `fps=12` to `fps=10` or `fps=8`.
- Reduce `max_colors=192` to `128`.
- Trim the input more aggressively with `trim=start=2:end=35`.
- Increase the speed factor from `/3` to `/4`.

To make the GIF easier to read:

- Reduce the speed factor from `/3` to `/2`.
- Raise the width to `scale=1080:-1`.
- Raise the frame rate to `fps=15`.

These changes improve legibility or smoothness at the cost of a larger file.
For a full-length version, MP4 or WebM is strongly preferred over GIF because it
supports far better compression and playback quality.

## Notes

- Capture `demo.html` directly instead of the modal in `index.html`; this avoids
  modal chrome and ensures the animation starts from a clean page load.
- Keep the HTTP server running throughout capture so all relative images load.
- Waiting for `networkidle` and the page's own image/font readiness preserves the
  intended animation timing.
- Stop the local HTTP server after capture with `Ctrl+C`.
