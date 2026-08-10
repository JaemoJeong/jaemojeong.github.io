# SCoPE: Training-Free Audio-Visual Event Perception via Sparse Cross-Modal Prior Exchange

**Demo:** <https://jaemojeong.github.io/>

The default curated tab presents eight LLP examples with frozen Dense and
SCoPE predictions. Videos are streamed through YouTube's privacy-enhanced
embed; this repository does not distribute the source media. The two paper
figures, prediction payload, styles, and JavaScript are embedded in
`index.html`.

The separate **Streaming replay** tab is an exploratory causal adaptation that
was not evaluated in the paper. It uses three longer LFAV test videos and
reveals official strong audio/visual annotations alongside frozen CLIP + CLAP
Dense and prefix-causal SCoPE outputs after each completed second. The outputs
use the frozen ESC-50 audio and MS-COCO train2017 visual centering references
and are computed in advance; no model, capture pipeline, or inference service
runs in the browser. The active 10-second block fills from left to right, while
the displayed label rows follow the latest completed second.

## Local preview

```bash
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/>.

The replay supports only the three precomputed LFAV examples included in the
page data. It does not accept arbitrary videos or claim live inference. Source
videos remain on YouTube and are loaded through the privacy-enhanced embed; raw
LFAV media is not redistributed by this repository.
