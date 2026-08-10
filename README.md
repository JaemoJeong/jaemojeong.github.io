# SCoPE: Training-Free Audio-Visual Event Perception via Sparse Cross-Modal Prior Exchange

Interactive demo for training-free, causal audio-visual event parsing on the LLP test split.

**Live demo:** <https://jaemojeong.github.io/>

The site is a single self-contained `index.html`: nine video examples, two
paper figures, audio, predictions, styles, and JavaScript are embedded. No
model runs in the browser and no external asset is fetched.

## Local preview

```bash
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/>.

## Data and media

Predictions are precomputed from frozen SCoPE experiment artifacts. Video
excerpts come from the LLP test split and originate from AudioSet/YouTube. Media
rights remain with their respective owners.
