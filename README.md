# SCoPE: Training-Free Audio-Visual Event Perception via Sparse Cross-Modal Prior Exchange

**Demo:** <https://jaemojeong.github.io/>

The default curated tab presents eight LLP examples with frozen Dense and
SCoPE predictions. Videos are streamed through YouTube's privacy-enhanced
embed; this repository does not distribute the source media. The two paper
figures, prediction payload, styles, and JavaScript are embedded in
`index.html`.

The separate **Streaming replay** tab is an exploratory causal adaptation that
was not evaluated in the paper. It uses three longer LFAV test videos and lets
the viewer switch between frozen **CLIP + CLAP** and **LanguageBind** outputs;
LanguageBind is the default. Official strong audio/visual annotations are
revealed alongside Dense, zero-latency SCoPE, and its one-second-latency
readout. SCoPE exchanges reliable evidence over a trailing 10-second causal
memory without periodic resets or future segments. Each backbone uses ESC-50
audio and MS-COCO train2017 visual centering encoded in its own feature space.
All outputs are computed in advance; no model, capture pipeline, or inference
service runs in the browser. One continuous full-clip timeline fills from left
to right. Each method keeps only its own current predictions and current
ground-truth labels, so irrelevant rows disappear as playback advances.

## Local preview

```bash
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/>.

The replay supports only the three precomputed LFAV examples included in the
page data: a cello/speech tutorial, a basketball/crowd sequence, and a
dog/car/speech sequence. Their full lengths are 129, 136, and 106 one-second
segments; only the prior memory is limited to 10 seconds. The page does not
accept arbitrary videos or claim live inference. Source videos remain on
YouTube and are loaded through the privacy-enhanced embed; raw LFAV media is not
redistributed by this repository.
