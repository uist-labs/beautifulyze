"""Decoder tests for load_audio().

librosa 1.0 dropped the audioread/ffmpeg fallback, leaving soundfile as its
only backend — and libsndfile cannot open mp4/m4a/aac containers. load_audio()
therefore falls back to ffmpeg itself. These tests cover both paths.

As elsewhere in the suite, every fixture is synthesized at run time so the
tests never depend on the gitignored, copyrighted audio.
"""
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

import beautifulyze as bz

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _write_tone(path, freq=440.0, seconds=2.0, sr=44100):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, 0.5 * np.sin(2 * np.pi * freq * t), sr)
    return sr


def _transcode(src, dst, *args):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, str(dst)],
        check=True,
    )


# ── soundfile path (formats libsndfile handles natively) ──────────
@pytest.mark.parametrize("suffix", [".wav", ".flac", ".ogg"])
def test_load_audio_reads_libsndfile_formats(tmp_path, suffix):
    wav = tmp_path / "tone.wav"
    sr = _write_tone(wav)
    target = wav.with_suffix(suffix)
    if suffix != ".wav":
        pytest.importorskip("soundfile")
        sf.write(target, sf.read(wav)[0], sr)

    y, out_sr = bz.load_audio(target)

    assert out_sr == sr
    assert y.dtype == np.float32
    assert y.ndim == 1 and len(y) > 0


# ── ffmpeg fallback (containers libsndfile refuses) ───────────────
@needs_ffmpeg
@pytest.mark.parametrize("suffix,codec", [(".m4a", "aac"), (".mp4", "aac")])
def test_load_audio_falls_back_to_ffmpeg(tmp_path, suffix, codec):
    wav = tmp_path / "tone.wav"
    sr = _write_tone(wav)
    encoded = tmp_path / f"tone{suffix}"
    _transcode(wav, encoded, "-c:a", codec, "-b:a", "128k")

    y, out_sr = bz.load_audio(encoded)

    # Decoded at the native rate, as a writable mono float32 waveform.
    assert out_sr == sr
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert y.flags.writeable          # analysis downstream writes in place
    # AAC pads the head/tail, so allow slack around the 2 s source.
    assert 1.8 * sr < len(y) < 2.4 * sr


@needs_ffmpeg
def test_ffmpeg_and_soundfile_agree(tmp_path):
    """The two decode paths must not disagree on the same input."""
    wav = tmp_path / "tone.wav"
    _write_tone(wav)

    via_sf, sr_sf = bz.load_audio(wav)        # soundfile
    via_ff, sr_ff = bz._ffmpeg_load(wav)      # ffmpeg

    assert sr_sf == sr_ff
    assert len(via_sf) == len(via_ff)
    assert np.abs(via_sf - via_ff).max() == 0.0


@needs_ffmpeg
def test_load_audio_survives_a_lossy_round_trip(tmp_path):
    """A tone encoded to AAC still reads as concert A after the fallback."""
    wav = tmp_path / "tone.wav"
    _write_tone(wav, seconds=3.0)
    m4a = tmp_path / "tone.m4a"
    _transcode(wav, m4a, "-c:a", "aac", "-b:a", "192k")

    digest = bz.analyze(m4a, output=tmp_path / "tone.png", n_mels=64)

    assert digest["key"]["tonal_center"] == "A"


# ── error paths stay actionable ───────────────────────────────────
def test_load_audio_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        bz.load_audio(tmp_path / "nope.mp3")


@needs_ffmpeg
def test_load_audio_undecodable_file_raises_runtime_error(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not audio" * 100)

    # RuntimeError is what main() catches to keep batch runs going.
    with pytest.raises(RuntimeError, match="could not decode"):
        bz.load_audio(junk)
