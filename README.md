# SCoPE without the future

Interactive demo for training-free, causal audio-visual event parsing on the LLP test split.

**Live demo:** <https://jaemojeong.github.io/>

The page compares a fixed dense cosine readout with SCoPE at zero latency and with a one-second local-persistence readout. All aggregate results cover the full 1,109-video test split. The eight videos are deliberately selected qualitative examples and are labeled as such in the page.

The site is a single self-contained `index.html`: videos, audio, predictions, styles, and JavaScript are embedded. No model runs in the browser and no external asset is fetched.

## Local preview

```bash
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/>.

## Data and media

Predictions are precomputed from frozen SCoPE experiment artifacts. Video excerpts come from the LLP test split, whose source videos originate from AudioSet/YouTube. The excerpts remain the property of their respective uploaders and are included only for research demonstration.
