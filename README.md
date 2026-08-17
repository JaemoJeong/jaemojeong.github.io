# SCoPE: Training-Free Audio-Visual Event Perception via Sparse Cross-Modal Prior Exchange

**Demo:** <https://jaemojeong.github.io/>

The default curated tab presents eight LLP examples with frozen Dense and
SCoPE predictions. Videos are streamed through YouTube's privacy-enhanced
embed; this repository does not distribute the source media. The two paper
figures, prediction payload, styles, and JavaScript are embedded in
`index.html`.

The separate **Streaming replay** tab is an exploratory causal adaptation that
was not evaluated in the paper. The default CLIP + CLAP view opens with one
continuous 25-second LFAV excerpt that alternates between shofar and speech
five times without overlapping labels. Its fused audio-visual SCoPE readout is
shown alongside the official ground truth and the frozen dense baseline. The
causal state is reset at the excerpt boundary, so no earlier or future segment
is used. Four longer LFAV examples remain available for both CLIP + CLAP and
LanguageBind.

Official strong annotations are revealed alongside Dense, zero-latency SCoPE,
and the one-second-latency readout. SCoPE exchanges reliable evidence over a
trailing 10-second causal memory without future segments. Each backbone uses
ESC-50 audio and MS-COCO train2017 visual centering encoded in its own feature
space. All outputs are computed in advance; no model, capture pipeline, or
inference service runs in the browser.

## Local preview

```bash
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/>.

The page does not accept arbitrary videos or claim live inference. Source
videos remain on YouTube and are loaded through the privacy-enhanced embed; raw
LFAV media is not redistributed by this repository.
