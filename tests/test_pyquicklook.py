"""Tests that need no audio hardware or display, so they run on any OS."""
import types

import numpy as np
import pytest
from scipy.io import wavfile

import pyquicklook as pq

FS, N = 48000, 4800


def tone(f, i_amp=1.0, q_amp=1.0, n=N, fs=FS, dc=(0.0, 0.0)):
    """Stereo block (left = Q, right = I) of a complex exponential at f Hz."""
    t = np.arange(n) / fs
    i = i_amp * np.cos(2 * np.pi * f * t) + dc[0]
    q = q_amp * np.sin(2 * np.pi * f * t) + dc[1]
    return np.column_stack([q, i])


def peak(spec, fs=FS, n=N):
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    k = int(np.argmax(spec))
    return freqs[k], spec[k]


def level_at(spec, f, fs=FS, n=N):
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1 / fs))
    return spec[np.argmin(np.abs(freqs - f))]


@pytest.mark.parametrize("f", [-3000.0, 1250.0])
def test_sign_and_dbfs(f):
    _d, spec, _s = pq.process_block(tone(f), np.hanning(N))
    f_pk, db = peak(spec)
    assert f_pk == f
    assert db == pytest.approx(0.0, abs=0.05)          # full-scale tone = 0 dBFS
    assert level_at(spec, -f) < -100                   # no image


def test_swap_iq_mirrors():
    _d, spec, _s = pq.process_block(tone(-3000), np.hanning(N), swap_iq=True)
    assert peak(spec)[0] == 3000


def test_dc_removed():
    d, _spec, _s = pq.process_block(tone(1000, dc=(0.1, -0.05)), np.hanning(N))
    assert abs(d.mean()) < 1e-12


def test_iq_balance():
    blk = tone(2000, i_amp=0.5)
    _d, raw, _ = pq.process_block(blk, np.hanning(N), balance=False)
    # gain error g gives an image at 20*log10((1-g)/(1+g)) relative to the tone
    assert level_at(raw, -2000) - level_at(raw, 2000) == pytest.approx(
        20 * np.log10(0.5 / 1.5), abs=0.1)
    _d, bal, scale = pq.process_block(blk, np.hanning(N))
    assert scale == pytest.approx(2.0, rel=1e-3)
    assert level_at(bal, -2000) < -100


def test_silent_block_is_safe():
    d, spec, scale = pq.process_block(np.zeros((N, 2)), np.hanning(N))
    assert scale == 1.0 and np.all(spec == pq.DB_FLOOR)


def test_waterfall_keeps_narrow_tones():
    fs, n = 96000, 9600                                 # more bins than columns
    ql = pq.QuickLook(fs, n, False, True, (-120, 0), 10)
    assert ql.water.shape[1] <= pq.WATERFALL_MAX_COLS
    ql.update([tone(2440, n=n, fs=fs) * 0.5])
    assert ql.water[0].max() == pytest.approx(-6.0, abs=0.1)


def test_no_input_detection():
    ql = pq.QuickLook(FS, N, False, True, (-120, 0), 10)
    ql.update([np.zeros((N, 2))] * int(pq.SILENCE_WARN_S * FS / N))
    assert ql.no_input and "NO INPUT" in ql.fig._suptitle.get_text()
    ql.update([tone(1000) * 1e-4])
    assert not ql.no_input


@pytest.mark.parametrize("dtype,scale", [(np.int16, 32767), (np.float32, 1.0),
                                         (np.uint8, None)])
def test_wav_formats(tmp_path, dtype, scale):
    x = tone(1000) * 0.5
    data = (x * 127 + 128).astype(np.uint8) if scale is None else (x * scale).astype(dtype)
    p = tmp_path / "x.wav"
    wavfile.write(p, FS, data)
    src = pq.FileSource.from_wav(str(p), lambda fs: pq._block_size(fs, 10))
    assert src.fs == FS and src.block_size == N
    assert np.abs(src.data).max() == pytest.approx(0.5, abs=0.01)


def test_mono_wav_is_i_only(tmp_path):
    p = tmp_path / "mono.wav"
    wavfile.write(p, FS, (np.cos(2 * np.pi * 1000 * np.arange(N) / FS) * 16384)
                  .astype(np.int16))
    src = pq.FileSource.from_wav(str(p), lambda fs: N)
    assert src.mono and src.data.shape == (N, 2)
    assert not np.any(src.data[:, 0])                  # Q = 0
    _d, spec, scale = pq.process_block(src.data, np.hanning(N), mono=True)
    assert scale == 1.0                                # nothing to balance against
    freqs = np.fft.rfftfreq(N, 1 / FS)
    assert len(spec) == len(freqs)                     # one-sided
    k = int(np.argmax(spec))
    assert freqs[k] == 1000
    assert spec[k] == pytest.approx(20 * np.log10(0.5), abs=0.05)   # amplitude 0.5


@pytest.mark.parametrize("swap", [False, True])
def test_mono_full_scale_sine_is_0dbfs(swap):
    x = np.sin(2 * np.pi * 3000 * np.arange(N) / FS)
    _d, spec, _s = pq.process_block(pq.mono_as_stereo(x), np.hanning(N),
                                    swap_iq=swap, mono=True)
    freqs = np.fft.rfftfreq(N, 1 / FS)
    assert freqs[np.argmax(spec)] == 3000              # --swap-iq doesn't apply
    assert spec.max() == pytest.approx(0.0, abs=0.05)


def test_mono_display_is_one_sided():
    ql = pq.QuickLook(FS, N, False, True, (-120, 0), 10, mono=True)
    assert ql.ax_spec.get_xlim() == (0, FS / 2)
    assert ql.freqs.min() == 0 and len(ql.line_spec.get_xdata()) == N // 2 + 1
    ql.update([pq.mono_as_stereo(0.5 * np.cos(2 * np.pi * 2000 * np.arange(N) / FS))])
    assert ql.peak_text.get_text() == "2000 Hz\n-6.0 dBFS"
    assert ql.water[0].max() == pytest.approx(-6.0, abs=0.1)
    assert "MONO INPUT" in ql.fig._suptitle.get_text()
    assert [l.get_label() for l in ql.ax_time.get_lines()] == ["mono input"]


def test_mono_as_stereo_shapes():
    for x in (np.ones(8), np.ones((8, 1))):
        st = pq.mono_as_stereo(x)
        assert st.shape == (8, 2) and np.all(st[:, 0] == 0) and np.all(st[:, 1] == 1)


def test_cli_mono_file_warns(tmp_path, capsys):
    wav, png = tmp_path / "mono.wav", tmp_path / "out.png"
    wavfile.write(wav, FS, np.tile(tone(1000)[:, 1], 4).astype(np.float32) * 0.5)
    assert pq.main(["--file", str(wav), "--save", str(png), "--duration", "0.3"]) == 0
    assert "is mono" in capsys.readouterr().err


def test_cli_save(tmp_path, capsys):
    wav, png = tmp_path / "in.wav", tmp_path / "out.png"
    wavfile.write(wav, FS, (np.tile(tone(-1500), (4, 1)) * 16000).astype(np.int16))
    assert pq.main(["--file", str(wav), "--save", str(png), "--duration", "0.5"]) == 0
    assert png.stat().st_size > 10000
    assert "Saved" in capsys.readouterr().out


def test_cli_without_gui_explains(tmp_path, capsys):
    wav = tmp_path / "in.wav"
    wavfile.write(wav, FS, np.zeros((N, 2), np.int16))
    assert pq.main(["--file", str(wav)]) == 1           # Agg: no window possible
    assert "--save" in capsys.readouterr().err


class FakeSD:
    """Just enough of sounddevice to test device selection: a Windows-style
    list where the same card appears under several host APIs."""

    APIS = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    DEVS = [
        {"name": "Microphone (Realtek)", "hostapi": 0, "max_input_channels": 1,
         "default_samplerate": 44100.0},
        {"name": "Line In (Sound Blaster X4)", "hostapi": 0, "max_input_channels": 2,
         "default_samplerate": 44100.0},
        {"name": "Speakers", "hostapi": 0, "max_input_channels": 0,
         "default_samplerate": 44100.0},
        {"name": "Line In (Sound Blaster X4)", "hostapi": 1, "max_input_channels": 2,
         "default_samplerate": 48000.0},
    ]
    default = types.SimpleNamespace(device=[0, 2])

    def query_hostapis(self):
        return self.APIS

    def query_devices(self, idx=None, kind=None):
        if idx is None and kind is None:
            return self.DEVS
        return self.DEVS[self.default.device[0] if idx is None else idx]

    def check_input_settings(self, device, samplerate, channels):
        dev = self.query_devices(device, "input")
        if channels > dev["max_input_channels"]:
            raise RuntimeError("Invalid number of channels")
        if dev["hostapi"] == 1 and samplerate != dev["default_samplerate"]:
            raise RuntimeError("Invalid sample rate")   # WASAPI shared mode


@pytest.fixture
def fake_sd(monkeypatch):
    monkeypatch.setattr(pq, "_sounddevice", lambda: FakeSD())


def test_device_by_name_and_hostapi(fake_sd):
    assert pq._select_device("blaster", 48000) == (1, 2)      # first that works
    assert pq._select_device("wasapi", 48000) == (3, 2)
    assert pq._select_device("3", 48000) == (3, 2)


def test_device_skips_unusable_matches(fake_sd):
    assert pq._select_device("blaster", 44100) == (1, 2)
    with pytest.raises(ValueError, match="record at 44100"):
        pq._select_device("wasapi", 44100)


def test_mono_mic_used_with_one_channel(fake_sd):
    assert pq._select_device(None, 48000) == (None, 1)        # default = mono mic
    assert pq._select_device("realtek", 48000) == (0, 1)


def test_stereo_preferred_over_mono(fake_sd):
    # "MME" matches the mono mic first, but the stereo line in wins
    assert pq._select_device("mme", 48000) == (1, 2)


def test_list_devices(fake_sd, capsys):
    pq._list_devices()
    out = capsys.readouterr().out
    assert "*[0] Microphone (Realtek) [MME]" in out
    assert "[3] Line In (Sound Blaster X4) [Windows WASAPI]" in out
    assert "Speakers" not in out
