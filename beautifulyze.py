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
GLOBAL_DB_REF = 1.0      # Fixed dB reference across all songs (not per-song max)
HOP_LENGTH = 256
N_MELS = 256
HPSS_MARGIN = 3.0
OUTPUT_DPI = 180
DURATION_NORM = True     # Normalize x-axis to 0.0–1.0 for cross-song comparison

AUDIO_EXTS = {
    ".wav", ".flac", ".mp3", ".mp4", ".m4a", ".aac",
    ".ogg", ".opus", ".aiff", ".aif", ".wma",
}

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl–Schmuckler key profiles (tonic-relative weights, index 0 = tonic).
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


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


def load_audio(path):
    """Decode any ffmpeg-readable audio file to a mono-ish float waveform.

    Raises actionable errors a tired operator can act on immediately.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        with warnings.catch_warnings():
            # librosa routes compressed formats through audioread/ffmpeg and
            # emits a 1.0-deprecation notice; it still works on 0.11. Hush it.
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=None)
    except Exception as exc:  # noqa: BLE001 — re-raised with guidance below
        raise RuntimeError(
            f"Could not decode '{path}'. Compressed formats (mp3/mp4/m4a/…) "
            f"require ffmpeg on your PATH — install it and try again.\n"
            f"  Original error: {exc}"
        ) from exc
    return y, int(sr)


def extract_features(
    y, sr, *, hop_length=HOP_LENGTH, n_mels=N_MELS, hpss_margin=HPSS_MARGIN
):
    """Run the full feature pipeline once and hand back a Features bundle."""
    duration = float(librosa.get_duration(y=y, sr=sr))
    y_h, y_p = librosa.effects.hpss(y, margin=hpss_margin)

    def mel(signal):
        S = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_mels=n_mels, fmax=sr // 2, hop_length=hop_length
        )
        return librosa.power_to_db(S, ref=GLOBAL_DB_REF)

    chroma = librosa.feature.chroma_cqt(
        y=y_h, sr=sr, hop_length=hop_length, bins_per_octave=36
    )
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
        centroid=librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length
        )[0],
        bandwidth=librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop_length
        )[0],
        rms_db=librosa.power_to_db(rms ** 2, ref=GLOBAL_DB_REF),
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
        last = float(np.mean(c[-c.size // 3:]))
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


def build_caption(digest):
    """Stitch the digest fields into one honest, model-readable sentence."""
    key = digest["key"]
    parts = []
    tempo = digest.get("tempo_bpm")
    if tempo:
        parts.append(f"~{round(tempo)} BPM")
    parts.append(f"{key['tonal_center']} {key['mode']}")
    parts.append(_qual_dynamics(digest["dynamics"]["range_db"]))
    bright = digest["brightness"]
    parts.append(f"{_qual_brightness(bright['mean_hz'])} ({bright['trend']})")
    parts.append(_qual_harmonicity(digest["harmonicity"]))
    return ", ".join(parts) + "."


def compute_digest(feat):
    """Assemble the full digest dict (incl. caption) from extracted features."""
    tempo = float(
        np.atleast_1d(
            librosa.beat.beat_track(
                y=feat.y_p, sr=feat.sr, hop_length=feat.hop_length
            )[0]
        )[0]
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=feat.onset_p, sr=feat.sr, hop_length=feat.hop_length
    )
    onset_rate = len(onsets) / feat.duration if feat.duration else 0.0

    digest = {
        "duration_sec": round(feat.duration, 2),
        "sample_rate": int(feat.sr),
        "tempo_bpm": round(tempo, 1),
        "tempo_is_estimate": True,
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

    def specshow_kwargs(ax, cmap="magma"):
        # x-axis is always drawn as time, then relabeled to 0–1 when normalized.
        return dict(
            sr=sr, hop_length=hop, x_axis="time", y_axis="mel",
            fmax=sr // 2, ax=ax, cmap=cmap,
        )

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

    def norm_xticks(ax):
        # specshow draws the x-axis in seconds; relabel ticks as a 0–1 fraction
        # via a formatter (robust to autoscale, and warning-free unlike
        # set_xticklabels on a non-fixed locator).
        if not duration_norm:
            return
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x/duration:.2f}"))

    # ── ROW 0: Full Mel ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    img = librosa.display.specshow(feat.S_full, **specshow_kwargs(ax0))
    norm_xticks(ax0)
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
    norm_xticks(ax1)
    ax1.set_xlabel(x_label, color=WHITE)
    style(ax1, "Harmonic Component")

    ax2 = fig.add_subplot(gs[1, 1])
    librosa.display.specshow(feat.S_p, **specshow_kwargs(ax2, cmap="viridis"))
    norm_xticks(ax2)
    ax2.set_xlabel(x_label, color=WHITE)
    style(ax2, "Percussive Component")

    # ── ROW 2: Chromagram ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, :])
    librosa.display.specshow(
        feat.chroma, sr=sr, hop_length=hop, x_axis="time", y_axis="chroma",
        ax=ax3, cmap="inferno",
    )
    norm_xticks(ax3)
    ax3.set_xlabel(x_label, color=WHITE)
    style(ax3, "Chromagram — Harmonic Source Only (36 bins/octave)")

    # ── ROW 3: Onset strength (percussive behind, harmonic on top) ─
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.fill_between(times_onset, feat.onset_p, alpha=0.6, color="#00d4ff", label="Percussive")
    ax4.fill_between(times_onset, feat.onset_h, alpha=0.9, color="#ff6b35", label="Harmonic")
    ax4.legend(facecolor="#1a1a1a", labelcolor=WHITE, fontsize=8)
    ax4.set_xlabel(x_label, color=WHITE)
    style(ax4, "Onset Strength — Harmonic (top) vs Percussive")

    # ── ROW 3: Centroid + Bandwidth (clamped at zero) ────────────
    ax5 = fig.add_subplot(gs[3, 1])
    lower = np.maximum(feat.centroid - feat.bandwidth, 0)
    upper = feat.centroid + feat.bandwidth
    ax5.fill_between(times_feat, lower, upper, alpha=0.3, color="#00d4ff", label="Bandwidth")
    ax5.plot(times_feat, feat.centroid, color="#00d4ff", linewidth=0.8, label="Centroid")
    ax5.set_ylim(bottom=0)
    ax5.legend(facecolor="#1a1a1a", labelcolor=WHITE, fontsize=8)
    ax5.set_xlabel(x_label, color=WHITE)
    style(ax5, "Spectral Centroid + Bandwidth (timbral brightness)")

    # ── ROW 4: RMS Energy Envelope ───────────────────────────────
    ax6 = fig.add_subplot(gs[4, :])
    ax6.fill_between(times_feat, feat.rms_db, feat.rms_db.min(), alpha=0.8, color="#c084fc")
    ax6.plot(times_feat, feat.rms_db, color="#e0aaff", linewidth=0.8)
    ax6.set_xlabel(x_label, color=WHITE)
    ax6.set_ylabel("dB", color=WHITE)
    style(ax6, "RMS Energy Envelope (emotional intensity arc)")

    # ── RENDER ───────────────────────────────────────────────────
    fig.patch.set_facecolor(BG)
    suptitle = title.replace("_", " ")
    if digest:
        suptitle += f"\n{digest['caption']}"
    fig.suptitle(suptitle, color=WHITE, fontsize=14, y=0.998)
    plt.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────────────────────────
#  ORCHESTRATION
# ─────────────────────────────────────────────
def analyze(
    filepath,
    output=None,
    *,
    duration_norm=DURATION_NORM,
    write_digest=True,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    hpss_margin=HPSS_MARGIN,
    dpi=OUTPUT_DPI,
):
    """Render one audio file to a PNG (+ JSON digest). Returns the digest dict."""
    filepath = Path(filepath)
    output = Path(output) if output else Path(filepath.stem + ".png")

    print(f"Loading: {filepath.name}")
    y, sr = load_audio(filepath)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Duration: {duration:.1f}s  SR: {sr}Hz")

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
    render_figure(
        feat, output, title=filepath.stem, digest=digest,
        duration_norm=duration_norm, dpi=dpi,
    )
    print(f"  Saved: {output}")

    if digest is not None:
        json_path = output.with_suffix(".json")
        json_path.write_text(
            json.dumps({"source": filepath.name, **digest}, indent=2) + "\n"
        )
        print(f"  Saved: {json_path}")

    return digest


def collect_inputs(inputs):
    """Expand a list of files/directories into a sorted list of audio files."""
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(
                sorted(c for c in p.iterdir() if c.suffix.lower() in AUDIO_EXTS)
            )
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
    parser.add_argument(
        "inputs", nargs="+", help="Audio file(s) or directory(ies) to analyze"
    )
    parser.add_argument(
        "-o", "--output", help="Output PNG path (only valid with a single input)"
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Use wall-clock time on the x-axis instead of a 0–1 position",
    )
    parser.add_argument(
        "--no-digest", action="store_true", help="Skip the JSON digest"
    )
    parser.add_argument("--n-mels", type=int, default=N_MELS, help="Mel bands")
    parser.add_argument("--hop-length", type=int, default=HOP_LENGTH, help="STFT hop")
    parser.add_argument(
        "--hpss-margin", type=float, default=HPSS_MARGIN,
        help="Harmonic/percussive separation margin",
    )
    parser.add_argument("--dpi", type=int, default=OUTPUT_DPI, help="Output PNG DPI")
    args = parser.parse_args(argv)

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
                duration_norm=not args.no_normalize,
                write_digest=not args.no_digest,
                n_mels=args.n_mels,
                hop_length=args.hop_length,
                hpss_margin=args.hpss_margin,
                dpi=args.dpi,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
