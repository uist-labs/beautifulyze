# Changelog

All notable changes to `beautifulyze` will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Decoding mp4/m4a/aac on librosa ≥ 1.0.** librosa 1.0 removed its
  `audioread` fallback, leaving `soundfile` as the only backend — and
  libsndfile cannot open those containers, so every such file failed with
  `Format not recognised` no matter how healthy the local ffmpeg was. Since
  `requirements.txt` pins `librosa>=0.11`, any fresh install picked up 1.0 and
  hit this. `load_audio()` now tries soundfile first and falls back to decoding
  through ffmpeg directly, rather than by way of a removed dependency.
- The "install ffmpeg" error no longer fires when ffmpeg *is* installed; it is
  raised only when ffmpeg is genuinely missing from `PATH`. Undecodable files
  now report ffmpeg's own reason instead.

- **The harmonic/percussive panels no longer contradict the digest.**
  `extract_features()` already put all three mel panels on one absolute dB scale
  via `GLOBAL_DB_REF`, but `specshow` autoscaled each panel independently and
  discarded it — stretching a near-silent percussive residue to full brightness.
  A piece reporting `harmonicity: 0.997` rendered with a percussive panel that
  looked busier than its harmonic one. The panels now share one dB window, and
  the pair shares a colormap so they are perceptually comparable rather than
  merely numerically comparable.
- **Line panels no longer alias into solid blocks.** A full-length track puts
  40–90 feature frames into every horizontal pixel, so the onset, centroid, and
  RMS panels rendered as filled rectangles — the RMS panel in particular
  promised an "emotional intensity arc" and delivered a monolith. Each is now
  drawn as a per-pixel min/max envelope with a mean trend line, which holds its
  shape at any duration. Every example in `examples/` was affected.
- **The shared timeline is now actually shared.** The spectrogram panels drew a
  seconds axis and relabelled its ticks as fractions, landing on 0.00 / 0.17 /
  0.34 / … and stopping short of the right edge, while the line panels used a
  true 0–1 axis with 0.2 steps. Two rulers under panels meant to be read against
  each other. All six panels now carry identical ticks and span the full width.

### Added
- **`tempo_candidates` in the digest.** Beat trackers pick a metrical level and
  report it as fact; on slow material they routinely lock onto a subdivision.
  Ambient tracks designed around 60 BPM were reading as high as 169. The digest
  now lists the octaves the onset envelope actually supports, each with a
  relative `support` score, so a reader can weigh the alternative instead of
  inheriting one guess.
- **Captions that admit uncertainty.** The caption is burned into the PNG and is
  the first thing a model reads, but it stated a `confidence: 0.40` key as flat
  fact. Low-confidence keys are now hedged ("possibly C minor"), and an
  ambiguous tempo names its rival ("~167 BPM (or 83)") — preferring the half or
  double over a better-supported but more exotic ratio, since that is the
  reading a listener is most likely to dispute.
- Decoder tests covering both paths (soundfile and the ffmpeg fallback), that
  the two agree sample-for-sample, and that the error paths stay actionable.
  They synthesize and transcode their own fixtures, and skip where ffmpeg is
  unavailable.
- Tests for envelope reduction (peaks survive, troughs survive, ragged lengths
  still reach the end), tempo octave ranking, and caption hedging.
- **CI** (`.github/workflows/ci.yml`): ruff lint + format on every push and PR,
  and a test matrix over Python 3.9 / 3.11 / 3.13 with ffmpeg installed so the
  decoder tests run rather than skip. Dependencies are resolved fresh and
  uncached on purpose — a wheel cache is exactly what would have hidden the
  librosa 1.0 break — and a weekly cron surfaces the next upstream change here
  instead of on someone's new laptop.
- **ruff** configuration and a `.pre-commit-config.yaml`, with the tree swept
  once so formatting is settled rather than relitigated per review.

### Note
- The renders in `examples/` predate the legibility fixes above and still show
  the old per-panel autoscaling, mismatched axes, and aliased line panels. They
  need regenerating from the source audio, which is deliberately not in the repo.

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
