"""Tests for the render-legibility and estimate-honesty behaviour.

Three concerns, all of which were real defects found by reading the rendered
gallery rather than the code:

  * `envelope_reduce` — dense series must survive downsampling with their peaks
    intact, or every line panel aliases into a solid block on a real track.
  * `tempo_candidates` — a beat tracker's octave choice is an opinion, not a
    fact, so the rivals it rejected have to stay visible.
  * `build_caption` — the caption is burned into the PNG and read first by a
    model, so an uncertain reading must announce itself there.

All fixtures are synthesized; nothing here touches audio on disk.
"""

import numpy as np

import beautifulyze as bz


# ── envelope_reduce ───────────────────────────────────────────────
def test_envelope_reduce_passes_short_series_through():
    x = np.arange(50, dtype=float)
    y = np.sin(x)
    xr, lo, hi, mean = bz.envelope_reduce(x, y, columns=100)
    assert np.array_equal(xr, x)
    for band in (lo, hi, mean):
        assert np.array_equal(band, y)


def test_envelope_reduce_hits_the_requested_column_count():
    x = np.arange(10_000, dtype=float)
    y = np.random.default_rng(0).normal(size=10_000)
    xr, lo, hi, mean = bz.envelope_reduce(x, y, columns=250)
    assert len(xr) == len(lo) == len(hi) == len(mean) == 250


def test_envelope_reduce_preserves_an_isolated_spike():
    """The whole point: a one-frame peak must not be averaged out of existence."""
    y = np.zeros(10_000)
    y[7_777] = 42.0
    x = np.arange(10_000, dtype=float)

    _, _, hi, mean = bz.envelope_reduce(x, y, columns=200)

    assert hi.max() == 42.0  # survives in the max envelope
    assert mean.max() < 42.0  # and would have been diluted by a mean alone


def test_envelope_reduce_preserves_troughs_and_ordering():
    x = np.arange(6_000, dtype=float)
    y = np.linspace(-5.0, 5.0, 6_000)
    xr, lo, hi, _ = bz.envelope_reduce(x, y, columns=120)

    assert np.isclose(lo.min(), -5.0)
    assert np.isclose(hi.max(), 5.0)
    assert np.all(np.diff(xr) > 0)  # x stays monotonic


def test_envelope_reduce_reaches_the_end_of_a_ragged_series():
    """A length that doesn't divide evenly must still cover the final samples."""
    n = 10_007  # prime-ish: 100 columns leaves a remainder
    x = np.arange(n, dtype=float)
    y = np.zeros(n)
    y[-1] = 99.0  # spike lands in the discarded remainder

    xr, _, hi, _ = bz.envelope_reduce(x, y, columns=100)

    assert hi[-1] == 99.0
    assert xr[-1] >= n - (n // 100)


# ── tempo_candidates ──────────────────────────────────────────────
def _pulse_train(bpm, sr=22050, hop=256, seconds=30.0):
    """An onset envelope with a clean pulse every 60/bpm seconds."""
    frames = int(seconds * sr / hop)
    env = np.zeros(frames)
    step = (60.0 / bpm) * sr / hop
    env[np.round(np.arange(0, frames, step)).astype(int)[:-1]] = 1.0
    return env


def test_tempo_candidates_always_includes_the_primary():
    env = _pulse_train(120.0)
    out = bz.tempo_candidates(env, sr=22050, hop_length=256, primary=120.0)
    assert any(abs(c["bpm"] - 120.0) < 0.05 for c in out)


def test_tempo_candidates_surfaces_the_half_time_rival():
    """A 160 BPM pulse also supports 80 — the reading a listener may prefer."""
    env = _pulse_train(160.0)
    out = bz.tempo_candidates(env, sr=22050, hop_length=256, primary=160.0)
    assert any(abs(c["bpm"] - 80.0) < 1.0 for c in out)


def test_tempo_candidates_are_ranked_and_normalized():
    env = _pulse_train(100.0)
    out = bz.tempo_candidates(env, sr=22050, hop_length=256, primary=100.0)
    supports = [c["support"] for c in out]
    assert supports == sorted(supports, reverse=True)
    assert max(supports) == 1.0
    assert all(0.0 <= s <= 1.0 for s in supports)


def test_tempo_candidates_stay_musically_plausible():
    env = _pulse_train(90.0)
    out = bz.tempo_candidates(env, sr=22050, hop_length=256, primary=90.0)
    lo, hi = bz.TEMPO_PLAUSIBLE_BPM
    assert all(lo <= c["bpm"] <= hi for c in out)


# ── build_caption honesty ─────────────────────────────────────────
def _digest(*, confidence=0.9, candidates=None, tempo=120.0):
    d = {
        "tempo_bpm": tempo,
        "tempo_candidates": candidates
        if candidates is not None
        else [{"bpm": tempo, "support": 1.0}],
        "key": {"tonal_center": "C", "mode": "minor", "confidence": confidence},
        "dynamics": {"range_db": 12.0},
        "brightness": {"mean_hz": 1500.0, "trend": "rising"},
        "harmonicity": 0.9,
    }
    return d


def test_caption_hedges_a_low_confidence_key():
    caption = bz.build_caption(_digest(confidence=0.40))
    assert "possibly C minor" in caption


def test_caption_states_a_confident_key_plainly():
    caption = bz.build_caption(_digest(confidence=0.92))
    assert "C minor" in caption
    assert "possibly" not in caption


def test_caption_names_the_rival_tempo_octave():
    caption = bz.build_caption(
        _digest(
            tempo=166.7,
            candidates=[
                {"bpm": 166.7, "support": 1.0},
                {"bpm": 83.4, "support": 0.6},
            ],
        )
    )
    assert "167 BPM (or 83)" in caption


def test_caption_prefers_the_octave_over_a_better_supported_odd_ratio():
    """111 (2/3) may score higher, but 83 (1/2) is the reading users dispute."""
    caption = bz.build_caption(
        _digest(
            tempo=166.7,
            candidates=[
                {"bpm": 166.7, "support": 1.0},
                {"bpm": 111.1, "support": 0.76},
                {"bpm": 83.4, "support": 0.60},
            ],
        )
    )
    assert "(or 83)" in caption


def test_caption_omits_the_rival_when_the_tempo_is_unambiguous():
    caption = bz.build_caption(_digest(tempo=120.0))
    assert "120 BPM" in caption
    assert "(or" not in caption
