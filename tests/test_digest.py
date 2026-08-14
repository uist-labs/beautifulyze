"""Unit tests for the digest computation in beautifulyze.

These cover the pure, audio-free functions that turn extracted features into
the machine-readable summary. Run from the repo root with `python -m pytest`.
"""

import numpy as np
import pytest

import beautifulyze as bz


# ── estimate_key ──────────────────────────────────────────────────
def test_estimate_key_c_major_triad():
    # C-E-G (indices 0, 4, 7): a plain C major triad.
    chroma = np.zeros((12, 8))
    for pc in (0, 4, 7):
        chroma[pc, :] = 1.0
    key = bz.estimate_key(chroma)
    assert key["tonal_center"] == "C"
    assert key["mode"] == "major"
    assert 0.0 <= key["confidence"] <= 1.0


def test_estimate_key_c_minor_triad():
    # C-Eb-G (indices 0, 3, 7): minor third present, major third absent.
    chroma = np.zeros((12, 8))
    for pc in (0, 3, 7):
        chroma[pc, :] = 1.0
    key = bz.estimate_key(chroma)
    assert key["tonal_center"] == "C"
    assert key["mode"] == "minor"


def test_estimate_key_respects_pitch_class_rotation():
    # D-F#-A (indices 2, 6, 9): a D major triad. Pins the rotation/indexing.
    chroma = np.zeros((12, 8))
    for pc in (2, 6, 9):
        chroma[pc, :] = 1.0
    key = bz.estimate_key(chroma)
    assert key["tonal_center"] == "D"
    assert key["mode"] == "major"


# ── describe_brightness ───────────────────────────────────────────
def test_describe_brightness_rising():
    assert bz.describe_brightness(np.linspace(100, 1000, 50))["trend"] == "rising"


def test_describe_brightness_falling():
    assert bz.describe_brightness(np.linspace(1000, 100, 50))["trend"] == "falling"


def test_describe_brightness_steady_with_mean():
    out = bz.describe_brightness(np.full(50, 500.0))
    assert out["trend"] == "steady"
    assert out["mean_hz"] == pytest.approx(500.0)


# ── describe_dynamics ─────────────────────────────────────────────
def test_describe_dynamics_uses_robust_range():
    # 0..100: mean 50; p95-p5 = 95-5 = 90 (robust to silent edges).
    out = bz.describe_dynamics(np.arange(101, dtype=float))
    assert out["mean_db"] == pytest.approx(50.0)
    assert out["range_db"] == pytest.approx(90.0)


# ── harmonicity ───────────────────────────────────────────────────
def test_harmonicity_all_harmonic():
    assert bz.harmonicity(np.ones(100), np.zeros(100)) == pytest.approx(1.0)


def test_harmonicity_all_percussive():
    assert bz.harmonicity(np.zeros(100), np.ones(100)) == pytest.approx(0.0)


def test_harmonicity_balanced():
    assert bz.harmonicity(np.ones(100), np.ones(100)) == pytest.approx(0.5)


# ── build_caption ─────────────────────────────────────────────────
def _sample_digest():
    return {
        "duration_sec": 600.0,
        "tempo_bpm": 56.0,
        "key": {"tonal_center": "A", "mode": "minor", "confidence": 0.8},
        "dynamics": {"mean_db": -28.0, "range_db": 12.0},
        "brightness": {"mean_hz": 1200.0, "trend": "steady"},
        "harmonicity": 0.9,
        "onset_rate": 0.5,
    }


def test_build_caption_includes_key_facts():
    cap = bz.build_caption(_sample_digest())
    assert isinstance(cap, str) and cap.strip()
    assert "56" in cap  # tempo
    assert "A minor" in cap  # key + mode
    assert "narrow" in cap  # dynamic-range descriptor for range_db=12
    assert "harmonic" in cap.lower()  # harmonicity descriptor for 0.9
