# Frozen one-second GPU regression

`golden_F0Omj8D7rOg_seq0_prediction.json` is the canonical prediction-only
artifact for frame 0 and audio samples `[0,16000)` of the named LLP clip. It
contains no source media. Its SHA-256 is:

```text
07f99bfa9c0b7a738c463813a934eb8df0a2f8ba88f5870ccd883ac7bbd90e82
```

The artifact was reproduced byte-for-byte in five fresh processes and again
after the final provenance and session-hardening changes. The final run used
GPU 0 (NVIDIA TITAN RTX), Python 3.10.19, PyTorch 2.9.1+cu128, torchvision
0.24.1, torchaudio 2.9.1+cu128, NumPy 1.26.4, and the pinned Transformers
4.30.2 vendor tree.

Input digests:

- frame RGB bytes: `f9e206f1657a7c764725359184e0a961ffe5585cd22e68c59fe9d3ad33dee958`
- 16,000 float32 audio samples: `9aae3dcf6be35995b74781169c4e6444201881db0184c17d864e35b8e7d8e798`

The checkpoint, config, prototype, mean, LanguageBind source-tree, vendor-tree,
and imported SCoPE source hashes are enforced at production startup by
`scope_streaming/config.py` and `scope_streaming/engines.py`.

`synthetic_constant_rgb_zero_audio_prediction.json` is the rights-free
production startup sentinel. It uses a 48×64 RGB image filled with
`(30,120,220)` and 16,000 float32 zeros. Its SHA-256 is:

```text
95abcf9024c4b7012856d021769c8e1a8ff77ae50f980dd40b26dc7ea0e90fb6
```

Production recreates this input and compares the complete live Dense/SCoPE
output with the artifact after loading the real GPU model. The artifact was
byte-identical across three fresh processes before integration.
