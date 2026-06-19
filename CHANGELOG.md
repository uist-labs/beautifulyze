# Changelog

All notable changes to `beautifulyze` will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-19

First public release.

### Added
- Multi-panel audio visualizer: mel spectrogram, harmonic/percussive (HPSS)
  split, chromagram, harmonic-vs-percussive onset strength, spectral centroid +
  bandwidth, and an RMS energy arc — all on one shared, length-normalized
  timeline for cross-track comparison.
- **JSON digest** written beside every render: estimated tempo, key
  (Krumhansl–Schmuckler) with confidence, dynamics (mean + silence-robust
  range), brightness (centroid + trend), harmonicity, and onset rate, plus a
  one-line grounded `caption`. Estimated key and tempo are also surfaced on the
  figure itself.
- Proper CLI (`argparse`): positional files **or** directories, `-o/--output`,
  `--no-normalize`, `--no-digest`, and feature knobs (`--n-mels`,
  `--hop-length`, `--hpss-margin`, `--dpi`).
- **`--linear` reveal mode**: a single-panel linear-frequency STFT (the faithful
  view for images drawn into a spectrogram), with `--start`/`--end` windowing
  and `--n-fft`. Showcased by the Aphex Twin "[Equation]" hidden-face example.
- Installable as a `beautifulyze` console command (`pyproject.toml`) or runnable
  directly with `requirements.txt`.
- Actionable errors for a missing file or a missing ffmpeg decoder; batch runs
  continue past a failed track.
- Test suite: unit tests for the digest on synthetic signals, and an
  end-to-end regression test (440 Hz tone → reads as concert A).
- `examples/` gallery spanning minimalist, ambient, and beat-driven material.
