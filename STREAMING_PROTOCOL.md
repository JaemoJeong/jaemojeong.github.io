# SCoPE live streaming protocol

This page feature is an exploratory, LanguageBind-only streaming adaptation of
SCoPE. It is not an experiment reported in the paper and it does not compute or
display ground truth or evaluation metrics.

## Input and state

- A user explicitly shares a browser tab with audio enabled.
- The browser sends one current video frame and the preceding one second of
  mono audio for each sequence index `t = 0, 1, ...`.
- The inference service keeps no media after the request completes.
- A session lasts at most 60 seconds. Stop, capture termination, sequence gaps,
  or a seek/reset starts a new causal state.
- Requests are processed in sequence. If inference falls behind, the client
  does not build an unbounded queue.

The first result therefore needs one second of observation. This is chunk
acquisition time, not future lookahead. The UI reports acquisition, network,
and compute latency separately.

## Per-second inference

The service keeps LanguageBind resident on one GPU and freezes the 25 LLP event
prompts and all SCoPE parameters. At each second it:

1. encodes the current image and one-second audio chunk;
2. computes the Dense cosine readout;
3. solves the Stage-1 non-negative sparse decompositions for audio and image;
4. updates each source modality's reliability-weighted causal prefix prior;
5. exchanges the current prefix priors and solves the two Stage-2 target
   decompositions; and
6. applies the frozen zero-anchored largest-gap readout.

For modality `m`, class `c`, and current segment `t`, the streaming prior is

```text
P_m[t,c] = (1 / (t + 1)) * sum_{u=0..t} q_m[u] * s_m[u,c]
```

Only observations through `t` enter the result at `t`. No duration filter,
gap-closing step, or `t+1` confirmation is used in the live tab.

Production startup fails closed on the frozen checkpoints, configs,
prototypes, centering means, LanguageBind/vendor source manifests, imported
SCoPE source hashes, runtime versions, and a rights-free synthetic end-to-end
GPU golden prediction.

## UI contract

- The curated paper/demo examples remain unchanged in their own tab.
- The live tab is LanguageBind-only and shows neutral `Dense` and `SCoPE`
  predictions; it never colors predictions as correct or incorrect.
- The live tab says: `Exploratory streaming adaptation — not evaluated in the
  paper.`
- A visible status distinguishes capture, queued, processing, result, and error
  states.
- When no HTTPS inference endpoint is configured, the live controls remain
  disabled and explain that the backend is unavailable. No mock result may be
  shown on the public site.

## Deployment boundary

GitHub Pages hosts only the static client. A separate authenticated/rate-limited
HTTPS service performs inference. The public service must enforce the allowed
origin, one active stream per client, a 60-second session cap, strict payload
limits, session expiry, and no media persistence. It must not accept or fetch
arbitrary media URLs.
