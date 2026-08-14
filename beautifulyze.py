#!/usr/bin/env python3
"""beautifulyze — render audio into a picture an LLM can read.

One audio file in, two things out:

  * a multi-panel PNG (mel spectrogram, harmonic/percussive split, chromagram,
    onset strength, spectral centroid/bandwidth, RMS energy arc) on a shared,
    normalized timeline, and
  * a compact JSON digest (tempo, key, dynamics, brightness, harmonicity, …)
    plus a one-line caption that grounds the model's reading in hard numbers.

Feed the model a waveform and it learns nothing. Feed it *this* and it can
reason about structure, harmony, dynamics, and timbre at a glance.

Usage:
    beautifulyze song.flac
    beautifulyze track.mp3 -o out.png --no-normalize
    beautifulyze album/                 # render every audio file in a folder
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: render to file, never a display

import librosa
import librosa.display
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ─────────────────────────────────────────────
#  CONFIGURATION (defaults; all overridable via the CLI)
# ─────────────────────────────────────────────
GLOBAL_DB_REF = 1.0  # Fixed dB reference across all songs (not per-song max)
HOP_LENGTH = 256
N_MELS = 256
HPSS_MARGIN = 3.0
OUTPUT_DPI = 180
DURATION_NORM = True  # Normalize x-axis to 0.0–1.0 for cross-song comparison
N_FFT = 4096  # STFT window for --linear (linear-frequency) renders
SPEC_DB_FLOOR = 80.0  # Panels share one dB window: [peak - floor, peak]
ENVELOPE_COLUMNS = 1600  # Line panels reduce to this many min/max columns

# Tempo octaves worth reporting alongside the primary estimate. Beat trackers
# routinely lock onto a subdivision (or a multiple) of the pulse a listener
# taps, so a single number hides a real ambiguity — especially on slow music,
# where the tapped tempo is often half what the onset envelope peaks at.
TEMPO_OCTAVE_RATIOS = (0.25, 1 / 3, 0.5, 2 / 3, 1.5, 2.0, 3.0, 4.0)
TEMPO_CANDIDATE_FLOOR = 0.55  # Report a rival only at >=55% of the peak's support
TEMPO_PLAUSIBLE_BPM = (30.0, 300.0)
KEY_CONFIDENCE_HEDGE = 0.6  # Below this, the caption says "possibly X minor"

AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".mp3",
    ".mp4",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".aiff",
    ".aif",
    ".wma",
}

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl–Schmuckler key profiles (tonic-relative weights, index 0 = tonic).
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ─────────────────────────────────────────────
#  AUDIO + FEATURES
# ─────────────────────────────────────────────
@dataclass
class Features:
    """Everything the figure and the digest need, computed once per track."""

    y: np.ndarray
    sr: int
    duration: float
    hop_length: int
    y_h: np.ndarray
    y_p: np.ndarray
    S_full: np.ndarray
    S_h: np.ndarray
    S_p: np.ndarray
    chroma: np.ndarray
    onset_h: np.ndarray
    onset_p: np.ndarray
    centroid: np.ndarray
    bandwidth: np.ndarray
    rms_db: np.ndarray


def _ffprobe_sample_rate(path, default=44100):
    """Native sample rate of the first audio stream, or `default` if unknowable."""
    try:
        out = (
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate",
                    "-of",
                    "csv=p=0",
                    str(path),
                ],
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        return int(out) or default
    except (OSError, subprocess.CalledProcessError, ValueError):
        return default


def _ffmpeg_load(path):
    """Decode `path` to a mono float32 waveform by piping through ffmpeg.

    librosa dropped its audioread/ffmpeg fallback in 1.0, so soundfile is the
    only backend left — and libsndfile cannot open mp4/m4a/aac containers. This
    is that fallback, done directly and without the deprecated middleman.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"Could not decode '{path}'. Compressed formats (mp4/m4a/aac/…) are "
            f"decoded via ffmpeg, which is not on your PATH — install it and "
            f"try again."
        )
    sr = _ffprobe_sample_rate(path)
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-i",
                str(path),
                "-f",
                "f32le",  # raw little-endian float samples
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",  # downmix to mono, as librosa.load does
                "-ar",
                str(sr),
                "-",
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(
            f"ffmpeg could not decode '{path}'.\n  {detail[-1] if detail else 'no error output'}"
        ) from exc
    # frombuffer views immutable bytes; copy so downstream analysis can write.
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size == 0:
        raise RuntimeError(f"No decodable audio stream found in '{path}'.")
    return y, sr


def load_audio(path):
    """Decode any ffmpeg-readable audio file to a mono-ish float waveform.

    Tries soundfile (via librosa) first — it handles wav/flac/ogg/mp3 natively
    and quickly — then falls back to ffmpeg for everything libsndfile refuses,
    which is every mp4/m4a/aac file.

    Raises actionable errors a tired operator can act on immediately.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=None)
    except Exception:  # noqa: BLE001 — soundfile can't read it; try ffmpeg
        y, sr = _ffmpeg_load(path)
    return y, int(sr)


def envelope_reduce(x, y, columns=ENVELOPE_COLUMNS):
    """Reduce a dense series to per-column (min, max) pairs for plotting.

    A 10-minute track at hop 256 yields ~100k frames. Drawing that into ~1500
    pixels overplots roughly 70 frames per pixel, and every line panel collapses
    into a solid block — the shape the panel exists to show is lost to the
    rasterizer, not to the music.

    Bucketing into one column per pixel and keeping each bucket's extremes
    preserves peaks, troughs, and silhouette at any duration. Returns
    `(x_mid, y_min, y_max, y_mean)`; series already shorter than `columns` are
    returned unchanged (with min == max == mean) so short inputs are untouched.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(y) <= columns or columns < 1:
        return x, y, y, y

    # Trim to a whole number of buckets, then reduce each along axis 1.
    per = len(y) // columns
    used = per * columns
    buckets = y[:used].reshape(columns, per)
    x_mid = x[:used].reshape(columns, per).mean(axis=1)
    y_min = buckets.min(axis=1)
    y_max = buckets.max(axis=1)
    y_mean = buckets.mean(axis=1)

    # Fold any remainder into the last column so the series still reaches its end.
    if used < len(y):
        tail = y[used:]
        x_mid[-1] = x[used:].mean()
        y_min[-1] = min(y_min[-1], tail.min())
        y_max[-1] = max(y_max[-1], tail.max())
    return x_mid, y_min, y_max, y_mean


def slice_audio(y, sr, start=None, end=None):
    """Return the [start, end]-second slice of `y`. Either bound may be None."""
    if start is None and end is None:
        return y
    a = int(start * sr) if start is not None else 0
    b = int(end * sr) if end is not None else len(y)
    return y[a:b]


def extract_features(y, sr, *, hop_length=HOP_LENGTH, n_mels=N_MELS, hpss_margin=HPSS_MARGIN):
    """Run the full feature pipeline once and hand back a Features bundle."""
    duration = float(librosa.get_duration(y=y, sr=sr))
    y_h, y_p = librosa.effects.hpss(y, margin=hpss_margin)

    def mel(signal):
        S = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_mels=n_mels, fmax=sr // 2, hop_length=hop_length
        )
        return librosa.power_to_db(S, ref=GLOBAL_DB_REF)

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=hop_length, bins_per_octave=36)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    return Features(
        y=y,
        sr=sr,
        duration=duration,
        hop_length=hop_length,
        y_h=y_h,
        y_p=y_p,
        S_full=mel(y),
        S_h=mel(y_h),
        S_p=mel(y_p),
        chroma=chroma,
        onset_h=librosa.onset.onset_strength(y=y_h, sr=sr, hop_length=hop_length),
        onset_p=librosa.onset.onset_strength(y=y_p, sr=sr, hop_length=hop_length),
        centroid=librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0],
        bandwidth=librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0],
        rms_db=librosa.power_to_db(rms**2, ref=GLOBAL_DB_REF),
    )


# ─────────────────────────────────────────────
#  DIGEST — the LLM-facing summary (pure, testable)
# ─────────────────────────────────────────────
def estimate_key(chroma):
    """Best-fit key from a chromagram via Krumhansl–Schmuckler correlation.

    chroma: (12, T) with pitch-class order C, C#, …, B.
    Returns {'tonal_center', 'mode', 'confidence'} where confidence is the
    Pearson correlation of the winning profile (0..1).
    """
    profile = np.asarray(chroma, float).mean(axis=1)
    pc = profile - profile.mean()
    best_corr, best_tonic, best_mode = -2.0, 0, "major"
    for tonic in range(12):
        for mode, ks in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
            ref = np.roll(ks, tonic)
            ref = ref - ref.mean()
            denom = np.linalg.norm(pc) * np.linalg.norm(ref)
            corr = float(np.dot(pc, ref) / denom) if denom else 0.0
            if corr > best_corr:
                best_corr, best_tonic, best_mode = corr, tonic, mode
    return {
        "tonal_center": PITCH_CLASSES[best_tonic],
        "mode": best_mode,
        "confidence": round(max(0.0, best_corr), 3),
    }


def describe_brightness(centroid):
    """Mean spectral centroid (Hz) and whether it rises, falls, or holds."""
    c = np.asarray(centroid, float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return {"mean_hz": 0.0, "trend": "steady"}
    mean_hz = float(np.mean(c))
    trend = "steady"
    if c.size >= 6:
        first = float(np.mean(c[: c.size // 3]))
        last = float(np.mean(c[-c.size // 3 :]))
        rel = (last - first) / (abs(first) + 1e-9)
        trend = "rising" if rel > 0.10 else "falling" if rel < -0.10 else "steady"
    return {"mean_hz": round(mean_hz, 1), "trend": trend}


def describe_dynamics(rms_db):
    """Mean loudness and a silence-robust dynamic range (p95 − p5), in dB."""
    x = np.asarray(rms_db, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean_db": 0.0, "range_db": 0.0}
    p5, p95 = np.percentile(x, [5, 95])
    return {
        "mean_db": round(float(np.mean(x)), 1),
        "range_db": round(float(p95 - p5), 1),
    }


def harmonicity(y_h, y_p):
    """Fraction of energy in the harmonic component: 0 = percussive, 1 = tonal."""
    eh = float(np.sum(np.asarray(y_h, float) ** 2))
    ep = float(np.sum(np.asarray(y_p, float) ** 2))
    total = eh + ep
    return eh / total if total else 0.0


def _qual_dynamics(range_db):
    if range_db < 8:
        return "very narrow dynamic range"
    if range_db < 15:
        return "narrow dynamic range"
    if range_db < 25:
        return "moderate dynamic range"
    return "wide dynamic range"


def _qual_brightness(mean_hz):
    if mean_hz < 800:
        return "dark, warm timbre"
    if mean_hz < 2500:
        return "warm timbre"
    if mean_hz < 5000:
        return "bright timbre"
    return "very bright timbre"


def _qual_harmonicity(h):
    if h >= 0.66:
        return "strongly harmonic"
    if h <= 0.34:
        return "strongly percussive"
    return "balanced harmonic/percussive"


def tempo_candidates(onset_env, sr, hop_length, primary):
    """Rank the metrical levels the onset envelope actually supports.

    `primary` is whatever the beat tracker committed to. A tempogram scores how
    much periodic support each rival octave has; anything within
    TEMPO_CANDIDATE_FLOOR of the best is worth surfacing rather than silently
    discarding. Returns a list of `{"bpm": float, "support": float}`, strongest
    first, always including `primary`.
    """
    tgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    freqs = librosa.tempo_frequencies(tgram.shape[0], sr=sr, hop_length=hop_length)
    profile = np.nan_to_num(tgram.mean(axis=1))

    lo, hi = TEMPO_PLAUSIBLE_BPM

    def support(bpm):
        """Tempogram strength at `bpm`, via its nearest analysed lag."""
        if not lo <= bpm <= hi:
            return None
        idx = int(np.argmin(np.abs(freqs - bpm)))
        return float(profile[idx])

    scored = {}
    for ratio in (1.0, *TEMPO_OCTAVE_RATIOS):
        bpm = round(primary * ratio, 1)
        s = support(bpm)
        if s is not None:
            scored[bpm] = max(scored.get(bpm, 0.0), s)

    if not scored:
        return [{"bpm": round(primary, 1), "support": 1.0}]

    best = max(scored.values()) or 1.0
    ranked = sorted(
        ({"bpm": bpm, "support": round(s / best, 3)} for bpm, s in scored.items()),
        key=lambda c: c["support"],
        reverse=True,
    )
    keep = [c for c in ranked if c["support"] >= TEMPO_CANDIDATE_FLOOR]
    # The tracker's own answer always stays on the list, even if poorly supported.
    if not any(abs(c["bpm"] - round(primary, 1)) < 0.05 for c in keep):
        keep.append(next(c for c in ranked if abs(c["bpm"] - round(primary, 1)) < 0.05))
    return keep


def build_caption(digest):
    """Stitch the digest fields into one honest, model-readable sentence.

    "Honest" is load-bearing: the caption is what a model reads first and what
    is burned into the PNG, so an uncertain reading must *say* it is uncertain
    here — not only in a JSON field the image never shows.
    """
    key = digest["key"]
    parts = []
    tempo = digest.get("tempo_bpm")
    if tempo:
        # Surface one rival metrical level. A strict half/double is the reading
        # a listener is most likely to disagree with the tracker about, so it
        # wins over a better-supported but more exotic ratio (2/3, 3/2, …).
        rivals = [
            c["bpm"] for c in digest.get("tempo_candidates", []) if abs(c["bpm"] - tempo) >= 0.05
        ]
        octaves = [b for b in rivals if abs(b - tempo / 2) < 0.6 or abs(b - tempo * 2) < 0.6]
        rival = (octaves or rivals or [None])[0]
        parts.append(
            f"~{round(tempo)} BPM (or {round(rival)})" if rival else f"~{round(tempo)} BPM"
        )
    key_phrase = f"{key['tonal_center']} {key['mode']}"
    if key.get("confidence", 1.0) < KEY_CONFIDENCE_HEDGE:
        key_phrase = f"possibly {key_phrase}"
    parts.append(key_phrase)
    parts.append(_qual_dynamics(digest["dynamics"]["range_db"]))
    bright = digest["brightness"]
    parts.append(f"{_qual_brightness(bright['mean_hz'])} ({bright['trend']})")
    parts.append(_qual_harmonicity(digest["harmonicity"]))
    return ", ".join(parts) + "."


def compute_digest(feat):
    """Assemble the full digest dict (incl. caption) from extracted features."""
    tempo = float(
        np.atleast_1d(
            librosa.beat.beat_track(y=feat.y_p, sr=feat.sr, hop_length=feat.hop_length)[0]
        )[0]
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=feat.onset_p, sr=feat.sr, hop_length=feat.hop_length
    )
    onset_rate = len(onsets) / feat.duration if feat.duration else 0.0

    candidates = tempo_candidates(feat.onset_p, feat.sr, feat.hop_length, tempo)

    digest = {
        "duration_sec": round(feat.duration, 2),
        "sample_rate": int(feat.sr),
        "tempo_bpm": round(tempo, 1),
        "tempo_is_estimate": True,
        "tempo_candidates": candidates,
        "key": estimate_key(feat.chroma),
        "dynamics": describe_dynamics(feat.rms_db),
        "brightness": describe_brightness(feat.centroid),
        "harmonicity": round(harmonicity(feat.y_h, feat.y_p), 3),
        "onset_rate": round(onset_rate, 3),
    }
    digest["caption"] = build_caption(digest)
    return digest


# ─────────────────────────────────────────────
#  FIGURE
# ─────────────────────────────────────────────
BG = "#0a0a0a"
PANEL = "#111111"
SPINE = "#333333"
WHITE = "white"


def render_figure(feat, output, *, title, digest=None, duration_norm=True, dpi=OUTPUT_DPI):
    """Draw the seven-panel figure and save it to `output`."""
    sr, hop, duration = feat.sr, feat.hop_length, feat.duration

    def make_times(feature_array):
        t = librosa.times_like(feature_array, sr=sr, hop_length=hop)
        return t / duration if duration_norm else t

    times_onset = make_times(feat.onset_h)
    times_feat = make_times(feat.centroid)
    x_label = "Normalized Position (0=start, 1=end)" if duration_norm else "Time (s)"

    # One dB window for all three mel panels. extract_features() already puts
    # them on a common absolute scale via GLOBAL_DB_REF; letting specshow
    # autoscale each panel throws that away and stretches a near-silent
    # percussive residue to full brightness, so the picture contradicts the
    # harmonicity number sitting next to it.
    spec_vmax = float(max(feat.S_full.max(), feat.S_h.max(), feat.S_p.max()))
    spec_vmin = spec_vmax - SPEC_DB_FLOOR

    def specshow_kwargs(ax, cmap="magma"):
        return {
            "sr": sr,
            "hop_length": hop,
            "x_axis": "time",
            "y_axis": "mel",
            "fmax": sr // 2,
            "ax": ax,
            "cmap": cmap,
            "vmin": spec_vmin,
            "vmax": spec_vmax,
        }

    fig = plt.figure(figsize=(24, 22), facecolor=BG)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.3)

    def style(ax, panel_title):
        ax.set_title(panel_title, color=WHITE, fontsize=10)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)

    # Every panel gets the same six gridlines at the same fractions. The line
    # panels are plotted in 0–1 directly; the specshow panels are in seconds, so
    # their ticks are placed at the matching second offsets. Relabelling
    # whatever ticks specshow chose (the previous approach) left the two groups
    # on different rulers — 0.00/0.17/0.34/… against 0.0/0.2/0.4/… — and stopped
    # the top row short of the right edge, which quietly broke the cross-panel
    # comparison the shared timeline exists to support.
    TICK_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    def norm_xticks(ax, *, in_seconds):
        if not duration_norm:
            return
        span = duration if in_seconds else 1.0
        ax.set_xlim(0.0, span)
        ax.set_xticks([f * span for f in TICK_FRACTIONS])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x / span:.1f}"))

    # ── ROW 0: Full Mel ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    img = librosa.display.specshow(feat.S_full, **specshow_kwargs(ax0))
    norm_xticks(ax0, in_seconds=True)
    ax0.set_xlabel(x_label, color=WHITE)
    mel_title = f"Mel Spectrogram — Full Resolution  |  {title}  |  {duration:.1f}s"
    if digest:
        k = digest["key"]
        mel_title += f"  |  {k['tonal_center']} {k['mode']}  |  ~{round(digest['tempo_bpm'])} BPM"
    style(ax0, mel_title)
    fig.colorbar(img, ax=ax0, format="%+2.0f dB", pad=0.01).ax.yaxis.set_tick_params(
        color=WHITE, labelcolor=WHITE
    )

    # ── ROW 1: H/P Split ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    librosa.display.specshow(feat.S_h, **specshow_kwargs(ax1, cmap="magma"))
    norm_xticks(ax1, in_seconds=True)
    ax1.set_xlabel(x_label, color=WHITE)
    style(ax1, "Harmonic Component")

    ax2 = fig.add_subplot(gs[1, 1])
    # Same colormap as the harmonic panel, deliberately: two different ramps put
    # the pair on different perceptual scales even when the dB window matches,
    # and these two panels exist to be compared against each other.
    librosa.display.specshow(feat.S_p, **specshow_kwargs(ax2, cmap="magma"))
    norm_xticks(ax2, in_seconds=True)
    ax2.set_xlabel(x_label, color=WHITE)
    style(ax2, "Percussive Component")

    # ── ROW 2: Chromagram ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    librosa.display.specshow(
        feat.chroma,
        sr=sr,
        hop_length=hop,
        x_axis="time",
        y_axis="chroma",
        ax=ax3,
        cmap="inferno",
    )
    norm_xticks(ax3, in_seconds=True)
    ax3.set_xlabel(x_label, color=WHITE)
    style(ax3, "Chromagram — Harmonic Source Only (36 bins/octave)")

    # The line panels below are the ones that alias into solid blocks on
    # full-length tracks, so each is drawn as a peak envelope (the extremes per
    # pixel column) with a mean trend line over it. Half-width panels get half
    # the columns.
    half_cols = ENVELOPE_COLUMNS // 2

    # ── ROW 3: Onset strength (percussive behind, harmonic on top) ─
    ax4 = fig.add_subplot(gs[3, 0])
    xo_p, _, peak_p, _ = envelope_reduce(times_onset, feat.onset_p, half_cols)
    xo_h, _, peak_h, _ = envelope_reduce(times_onset, feat.onset_h, half_cols)
    # Peaks, not means: an onset envelope is *about* its spikes.
    ax4.fill_between(xo_p, peak_p, alpha=0.6, color="#00d4ff", label="Percussive")
    ax4.fill_between(xo_h, peak_h, alpha=0.9, color="#ff6b35", label="Harmonic")
    ax4.legend(facecolor="#1a1a1a", labelcolor=WHITE, fontsize=8)
    ax4.set_xlabel(x_label, color=WHITE)
    norm_xticks(ax4, in_seconds=False)
    style(ax4, "Onset Strength — Harmonic (front) vs Percussive")

    # ── ROW 3: Centroid + Bandwidth (clamped at zero) ────────────
    ax5 = fig.add_subplot(gs[3, 1])
    lower = np.maximum(feat.centroid - feat.bandwidth, 0)
    upper = feat.centroid + feat.bandwidth
    xb, lo_b, _, _ = envelope_reduce(times_feat, lower, half_cols)
    _, _, hi_b, _ = envelope_reduce(times_feat, upper, half_cols)
    xc, _, _, cen = envelope_reduce(times_feat, feat.centroid, half_cols)
    ax5.fill_between(xb, lo_b, hi_b, alpha=0.3, color="#00d4ff", label="Bandwidth")
    ax5.plot(xc, cen, color="#00d4ff", linewidth=0.8, label="Centroid")
    ax5.set_ylim(bottom=0)
    ax5.legend(facecolor="#1a1a1a", labelcolor=WHITE, fontsize=8)
    ax5.set_xlabel(x_label, color=WHITE)
    norm_xticks(ax5, in_seconds=False)
    style(ax5, "Spectral Centroid + Bandwidth (timbral brightness)")

    # ── ROW 4: RMS Energy Envelope ───────────────────────────────
    ax6 = fig.add_subplot(gs[4, :])
    xr, rms_lo, rms_hi, rms_mean = envelope_reduce(times_feat, feat.rms_db)
    floor = float(feat.rms_db.min())
    # Filling from the floor to the peak spans the track's whole dB range, which
    # on compressed material is a solid rectangle with the arc buried in it.
    # Keep that silhouette faint for the fade-in/out, and carry the actual
    # reading in the loud/quiet ribbon and the mean line drawn over it.
    ax6.fill_between(xr, floor, rms_hi, alpha=0.12, color="#c084fc")
    ax6.fill_between(xr, rms_lo, rms_hi, alpha=0.75, color="#c084fc")
    ax6.plot(xr, rms_mean, color="#f3e8ff", linewidth=1.2)
    ax6.set_xlabel(x_label, color=WHITE)
    ax6.set_ylabel("dB", color=WHITE)
    norm_xticks(ax6, in_seconds=False)
    style(ax6, "RMS Energy Envelope (emotional intensity arc)")

    # ── RENDER ───────────────────────────────────────────────────
    fig.patch.set_facecolor(BG)
    suptitle = title.replace("_", " ")
    if digest:
        suptitle += f"\n{digest['caption']}"
    fig.suptitle(suptitle, color=WHITE, fontsize=14, y=0.998)
    plt.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def render_linear(
    y, sr, output, *, title, n_fft=N_FFT, hop_length=HOP_LENGTH, dpi=OUTPUT_DPI, offset=0.0
):
    """Draw a single-panel linear-frequency STFT spectrogram and save it.

    Unlike the mel figure, the frequency axis here is *linear* — the faithful
    representation in which spatial content (e.g. images hidden in a track) is
    drawn. Pair with --start/--end to zoom in on a region.
    """
    S_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max
    )
    fig, ax = plt.subplots(figsize=(28, 11), facecolor=BG)
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
    if offset:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x + offset:.0f}s"))
    ax.set_xlabel("Time" + (f" (offset {offset:.0f}s)" if offset else " (s)"), color=WHITE)
    ax.set_ylabel("Hz", color=WHITE)
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    cbar.ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE)
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"{title.replace('_', ' ')} — linear STFT", color=WHITE, fontsize=14, y=0.98)
    plt.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────────────────────────
#  ORCHESTRATION
# ─────────────────────────────────────────────
def analyze(
    filepath,
    output=None,
    *,
    linear=False,
    start=None,
    end=None,
    duration_norm=DURATION_NORM,
    write_digest=True,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    hpss_margin=HPSS_MARGIN,
    n_fft=N_FFT,
    dpi=OUTPUT_DPI,
):
    """Render one audio file to a PNG (+ JSON digest). Returns the digest dict."""
    filepath = Path(filepath)
    output = Path(output) if output else Path(filepath.stem + ".png")

    print(f"Loading: {filepath.name}")
    y, sr = load_audio(filepath)
    y = slice_audio(y, sr, start, end)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Duration: {duration:.1f}s  SR: {sr}Hz")

    # Features feed the mel figure and/or the digest. In linear mode we only
    # need them if a digest was requested.
    feat = None
    if not linear or write_digest:
        print("  Extracting features (HPSS, mel, chroma, onsets)...")
        feat = extract_features(
            y, sr, hop_length=hop_length, n_mels=n_mels, hpss_margin=hpss_margin
        )

    digest = None
    if write_digest:
        print("  Computing digest...")
        digest = compute_digest(feat)
        print(f"  {digest['caption']}")

    print("  Rendering...")
    if linear:
        render_linear(
            y,
            sr,
            output,
            title=filepath.stem,
            n_fft=n_fft,
            hop_length=hop_length,
            dpi=dpi,
            offset=start or 0.0,
        )
    else:
        render_figure(
            feat,
            output,
            title=filepath.stem,
            digest=digest,
            duration_norm=duration_norm,
            dpi=dpi,
        )
    print(f"  Saved: {output}")

    if digest is not None:
        json_path = output.with_suffix(".json")
        json_path.write_text(json.dumps({"source": filepath.name, **digest}, indent=2) + "\n")
        print(f"  Saved: {json_path}")

    return digest


def collect_inputs(inputs):
    """Expand a list of files/directories into a sorted list of audio files."""
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(c for c in p.iterdir() if c.suffix.lower() in AUDIO_EXTS))
        else:
            files.append(p)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="beautifulyze",
        description="Render audio into a multi-panel picture an LLM can read, "
        "plus a JSON digest (tempo, key, dynamics, brightness, harmonicity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Audio file(s) or directory(ies) to analyze")
    parser.add_argument("-o", "--output", help="Output PNG path (only valid with a single input)")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Render a single-panel linear-frequency STFT instead of the mel "
        "figure — the faithful view for spotting images hidden in a track",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Trim to start at this time (seconds) before rendering",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="Trim to end at this time (seconds) before rendering",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Use wall-clock time on the x-axis instead of a 0–1 position",
    )
    parser.add_argument("--no-digest", action="store_true", help="Skip the JSON digest")
    parser.add_argument("--n-mels", type=int, default=N_MELS, help="Mel bands")
    parser.add_argument("--hop-length", type=int, default=HOP_LENGTH, help="STFT hop")
    parser.add_argument(
        "--hpss-margin",
        type=float,
        default=HPSS_MARGIN,
        help="Harmonic/percussive separation margin",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=N_FFT,
        help="STFT window for --linear",
    )
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI, help="Output PNG DPI")
    args = parser.parse_args(argv)

    if args.start is not None and args.end is not None and args.start >= args.end:
        parser.error("--start must be less than --end")

    files = collect_inputs(args.inputs)
    if not files:
        parser.error("no audio files found in the given input(s)")
    if args.output and len(files) != 1:
        parser.error("-o/--output is only valid with a single input file")

    failures = 0
    for f in files:
        try:
            analyze(
                f,
                output=args.output,
                linear=args.linear,
                start=args.start,
                end=args.end,
                duration_norm=not args.no_normalize,
                write_digest=not args.no_digest,
                n_mels=args.n_mels,
                hop_length=args.hop_length,
                hpss_margin=args.hpss_margin,
                n_fft=args.n_fft,
                dpi=args.dpi,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
