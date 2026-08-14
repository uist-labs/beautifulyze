"""End-to-end regression test for the analyze() pipeline.

Synthesizes a known signal (a 440 Hz tone = concert A) so the suite never
depends on the gitignored, copyrighted audio. Verifies the orchestration
writes a PNG + a well-formed JSON digest and that the digest reads the signal
correctly (key A, strongly harmonic).
"""

import json

import numpy as np
import soundfile as sf

import beautifulyze as bz


def _write_tone(path, freq=440.0, seconds=3.0, sr=22050):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, 0.5 * np.sin(2 * np.pi * freq * t), sr)


def test_analyze_writes_png_and_digest(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_tone(wav)
    png = tmp_path / "tone.png"

    digest = bz.analyze(wav, output=png, n_mels=64)

    # Artifacts on disk.
    assert png.exists() and png.stat().st_size > 0
    json_path = png.with_suffix(".json")
    assert json_path.exists()

    # JSON is well-formed and carries the expected shape.
    saved = json.loads(json_path.read_text())
    for field in (
        "source",
        "duration_sec",
        "tempo_bpm",
        "key",
        "dynamics",
        "brightness",
        "harmonicity",
        "onset_rate",
        "caption",
    ):
        assert field in saved
    assert saved["source"] == "tone.wav"
    assert saved["caption"] == digest["caption"]

    # The digest read the signal correctly: a 440 Hz tone is concert A and
    # is essentially all harmonic energy.
    assert saved["key"]["tonal_center"] == "A"
    assert saved["harmonicity"] > 0.8


def test_analyze_can_skip_digest(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_tone(wav)
    png = tmp_path / "tone.png"

    digest = bz.analyze(wav, output=png, write_digest=False, n_mels=64)

    assert digest is None
    assert png.exists()
    assert not png.with_suffix(".json").exists()


def test_collect_inputs_expands_directories(tmp_path):
    (tmp_path / "a.flac").touch()
    (tmp_path / "b.mp3").touch()
    (tmp_path / "notes.txt").touch()  # ignored: not audio

    found = bz.collect_inputs([tmp_path])

    assert [p.name for p in found] == ["a.flac", "b.mp3"]


# ── slice_audio (windowing for --start/--end / --linear reveal) ────
def test_slice_audio_both_bounds():
    sr = 10
    y = np.arange(100, dtype=float)
    out = bz.slice_audio(y, sr, start=2.0, end=5.0)
    assert len(out) == 30  # (5 - 2) s * 10 Hz
    assert out[0] == 20.0  # 2.0 s * 10 Hz


def test_slice_audio_start_only():
    sr = 10
    y = np.arange(100, dtype=float)
    out = bz.slice_audio(y, sr, start=2.0, end=None)
    assert len(out) == 80
    assert out[0] == 20.0


def test_slice_audio_end_only():
    sr = 10
    y = np.arange(100, dtype=float)
    out = bz.slice_audio(y, sr, start=None, end=5.0)
    assert len(out) == 50
    assert out[-1] == 49.0


def test_slice_audio_no_bounds_returns_input():
    y = np.arange(10, dtype=float)
    assert bz.slice_audio(y, 10, start=None, end=None) is y


def test_analyze_linear_writes_png(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_tone(wav, seconds=2.0)
    png = tmp_path / "tone_linear.png"

    bz.analyze(wav, output=png, linear=True, write_digest=False)

    assert png.exists() and png.stat().st_size > 0
