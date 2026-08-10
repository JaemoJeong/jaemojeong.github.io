# SCoPE: Training-Free Audio-Visual Event Perception via Sparse Cross-Modal Prior Exchange

**Demo:** <https://jaemojeong.github.io/>

The default curated tab presents eight LLP examples with frozen Dense and
SCoPE predictions. Videos are streamed through YouTube's privacy-enhanced
embed; this repository does not distribute the source media. The two paper
figures, prediction payload, styles, and JavaScript are embedded in
`index.html`.

The separate **Live streaming** tab is an exploratory causal adaptation that
was not evaluated in the paper. It captures a browser tab only after explicit
permission and sends one JPEG frame plus the preceding one-second audio chunk
to a LanguageBind/SCoPE inference API. It has no ground truth or metric view.
With the API meta tag left blank, live capture is disabled and the page never
generates mock or precomputed live results.

## Local preview

```bash
python3 -m http.server 8765
```

Then open <http://127.0.0.1:8765/>.

## Optional live backend

The prototype service and deployment contract are documented in
[`streaming_backend/README.md`](streaming_backend/README.md). It keeps the
LanguageBind encoder resident, accepts strictly ordered one-second chunks, and
uses only the received prefix for the SCoPE prior. Production requires an
HTTPS endpoint, frozen checkpoint/prototype/mean assets, an explicit CORS
allowlist, and external rate limiting. The frontend endpoint is configured via
the `scope-streaming-api` meta tag in `index.html`.
