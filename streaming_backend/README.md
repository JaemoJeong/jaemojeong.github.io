# SCoPE exploratory live-stream backend

This directory is an isolated backend prototype for a **new exploratory
streaming adaptation**. It is not an evaluated setting in the SCoPE paper and
does not modify the paper implementation or the static demo builder.

The server accepts one JPEG frame and one exactly one-second WAV chunk at a
time. It returns LanguageBind-only Dense and causal SCoPE predictions over the
canonical LLP vocabulary. There is deliberately no GT input, GT output, URL
fetcher, upload endpoint, or media cache.

## API

Start a session:

```http
POST /v1/sessions
Content-Type: application/json

{}
```

The response contains a random `session_id`, `next_sequence: 0`, the exact 25
class names, and the 60-second cap.

Submit each second in order:

```http
POST /v1/sessions/{session_id}/chunks
Content-Type: multipart/form-data

sequence = 0
frame    = image/jpeg
audio    = audio/wav
```

The WAV contract is strict: mono, 16-bit uncompressed PCM, 16 kHz, and exactly
16,000 samples. Responses contain continuous scores and 25 booleans for each
`audio`, `visual`, and `audio_visual` branch under both `dense` and `scope`, as
well as decode/encode/inference/total timing. Dense uses class-axis z-score +
sigmoid and strict `score > 0.85`. SCoPE uses raw nonnegative Stage-2 weights
and its zero-anchored largest-gap readout. No ground-truth field is emitted.

```http
POST   /v1/sessions/{session_id}/reset
DELETE /v1/sessions/{session_id}
GET    /v1/health
```

Duplicate, skipped, and reordered sequence numbers return HTTP 409. A session
expires after the configured idle TTL and cannot exceed 60 chunks.

## Exact causal SCoPE path

The production adapter imports, instead of copying, the configured public
SCoPE repository's:

- segment preparation, sparse decomposition, reconstruction quality, Dense
  normalization, and weighted target re-selection;
- final `P(c)=mean_t(q(t)s(t,c))` and direction-specific fixed-mean penalty;
- zero-anchored largest-gap and native audio-visual sparse fusion.

For second `t`, the final prior is evaluated only on numeric `q,s` values from
received seconds `0..t`. No global/full-video audio, max-over-future, temporal
cleanup, or lookahead is used. `prepare_modality` requires a video-shaped
placeholder; the current feature supplies that shape, but the adapter calls
only the three public **segment-only** functions. It never calls
`run_video_decomposition` or `run_stage12`.

The encoder is loaded once at process startup and remains resident. Frozen
LanguageBind-unit prototypes and external means are mandatory. Production
checks their frozen SHA-256 values and fails at startup if any is absent or
different; it never substitutes generated text
features, another backbone, another mean, CPU mock scores, or a downloadable
URL. Hugging Face/Transformers offline mode is forced before model loading, so
the configured cache must already contain both exact checkpoints.

After loading, production runs a rights-free constant-RGB/zero-audio
one-second sentinel through the actual GPU encoder and SCoPE path. Startup is
refused unless the complete Dense/SCoPE output matches the frozen artifact in
[`audit/`](audit/). This is a regression check, not an accuracy example.

The live adapter is closed to the canonical LLP-25 vocabulary. Its fixed
centering means were estimated from the LLP test-domain population, and a
one-second audio chunk is padded by the frozen LanguageBind audio transform.
This exploratory demo therefore is not an open-vocabulary or cross-dataset
streaming evaluation.

## Run

Install the small web layer, then use the pinned LanguageBind/SCoPE Python
3.10.19 environment that produced the frozen assets:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn scope_streaming.app:app --host 127.0.0.1 --port 8000
```

Export the values from `.env` through the deployment secret/config system.
Production defaults to `languagebind`; `SCOPE_DEVICE=cuda:3` is rejected.
Because GitHub Pages is HTTPS, a deployed frontend must point to an HTTPS API
and its exact origin must appear in `SCOPE_CORS_ORIGINS`.

For backend contract tests only, explicitly opt into deterministic mock mode:

```bash
SCOPE_STREAMING_MODE=mock uvicorn scope_streaming.app:app --port 8000
```

Mock responses are labeled as mock and are not scientific predictions.
The public frontend rejects mock sessions and accepts only an HTTPS production
LanguageBind endpoint, so this plain-HTTP command is not a live-UI preview.

## Security and retention

- Request bodies are bounded while streaming into memory; multipart parsing is
  also in-memory, avoiding framework upload spooling.
- JPEG dimensions/bytes and WAV encoding/bytes are checked before inference.
- Only an explicit CORS allowlist is accepted; wildcard origins are rejected.
- The service has no remote-URL input and writes no media to disk.
- Session state stores only numeric sparse evidence for at most 60 seconds.
- Model execution is serialized to protect the persistent GPU model.

For an internet deployment, put this behind TLS plus a reverse proxy with a
matching body limit, authentication/admission control, per-client rate and
concurrency limits, and an absolute request timeout. CORS is not
authorization. The in-process session store is a
single-worker prototype; multi-worker deployment requires sticky routing or a
shared state layer.

## Tests

Tests use only the explicit mock and never load LanguageBind, checkpoints, or a
GPU:

```bash
python -m unittest discover -s tests -v
```

The frozen prediction-only GPU regression and its reproducibility record are
in [`audit/`](audit/). It contains hashes and scores, not source media.
