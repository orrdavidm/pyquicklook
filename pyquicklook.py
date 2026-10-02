# pyquicklook.py
"""
Real-time I/Q quicklook: a Python port of ``quicklook_complex.m``.

A stereo sound-card capture is treated as a complex baseband signal
(left = Q/imaginary, right = I/real, matching the sample bitfile and the ZED
playback path) and shown as:

  * Time domain   - I and Q vs. time
  * Spectrum      - two-sided (-fs/2 .. +fs/2) FFT magnitude in dBFS, with the
                    strongest bin labelled
  * Waterfall     - spectrum history, sharing the spectrum's frequency axis

Processing per block mirrors the MATLAB script: remove DC, correct I/Q gain
imbalance, Hann window, FFT, fftshift.  Block length is fs / resolution, so
``--resolution`` is the FFT bin spacing just like MATLAB's
``frequency_resolution`` argument.

Improvements over the MATLAB version:
  * dBFS scaling (a full-scale complex tone reads 0 dB regardless of FFT size)
  * I/Q balance uses RMS rather than mean absolute value
  * waterfall display, device selection, --swap-iq, and --file playback of a
    recorded .wav so the tool can be used without a sound card

Close the plot window to stop.
"""

import argparse
import platform
import queue
import sys
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

DB_FLOOR = -140.0  # dBFS value used in place of log(0)
WATERFALL_MAX_COLS = 512   # wider spectra are max-pooled so narrow tones stay visible
SILENCE_WARN_S = 2.0       # this long of exact digital zeros means no input at all


# ---------------------------------------------------------------------------
# Signal processing (no audio or plotting dependencies, so it is testable)
# ---------------------------------------------------------------------------
def to_complex(stereo, swap_iq=False):
    """Combine a (frames, 2) stereo block into a complex I/Q vector.

    Left channel is Q (imaginary) and right channel is I (real), as in the
    MATLAB script.  ``swap_iq`` reverses that for boards/cables wired the
    other way round.
    """
    left = stereo[:, 0].astype(np.float64)
    right = stereo[:, 1].astype(np.float64)
    if swap_iq:
        left, right = right, left
    return right + 1j * left


def remove_dc(d):
    """Remove any DC offset contributed by the sound card."""
    return d - np.mean(d)


def iq_balance(d):
    """Scale I so its RMS matches Q, compensating for gain imbalance in the
    PC recording path or ZED playback path.

    Returns the corrected signal and the scale factor applied to I.
    """
    i_rms = np.sqrt(np.mean(d.real ** 2))
    q_rms = np.sqrt(np.mean(d.imag ** 2))
    if i_rms == 0 or q_rms == 0:
        return d, 1.0
    scale = q_rms / i_rms
    return d.real * scale + 1j * d.imag, scale


def spectrum_dbfs(d, window):
    """Two-sided, fftshift-ed magnitude spectrum in dBFS.

    Normalising by ``sum(window)`` makes a full-scale complex exponential
    read 0 dBFS independent of the FFT length.
    """
    mag = np.abs(np.fft.fftshift(np.fft.fft(d * window))) / np.sum(window)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(mag)
    return np.maximum(db, DB_FLOOR)


def process_block(stereo, window, swap_iq=False, balance=True):
    """Run one block through the MATLAB processing chain.

    Returns ``(time_signal, spectrum_db, i_scale)``.  As in MATLAB, the time
    signal is shown after DC removal but *before* I/Q balancing, so any
    imbalance is still visible in the time plot.
    """
    d = remove_dc(to_complex(stereo, swap_iq))
    balanced, scale = iq_balance(d) if balance else (d, 1.0)
    return d, spectrum_dbfs(balanced, window), scale


# ---------------------------------------------------------------------------
# Block sources
# ---------------------------------------------------------------------------
def _sounddevice():
    """Import sounddevice, explaining how to get PortAudio if it is missing.

    The macOS and Windows sounddevice wheels bundle PortAudio; on Linux it
    comes from the distribution.
    """
    try:
        import sounddevice
    except OSError as e:
        if platform.system() == "Linux":
            hint = ("install PortAudio: 'sudo apt install libportaudio2' "
                    "(Debian/Ubuntu), 'sudo dnf install portaudio' (Fedora) or "
                    "'sudo pacman -S portaudio' (Arch)")
        else:
            hint = "reinstall it: 'pip install --force-reinstall sounddevice'"
        raise RuntimeError(f"sounddevice could not load PortAudio ({e}); {hint}") from e
    return sounddevice


class LiveSource:
    """Stereo blocks from a sound card via sounddevice.

    The audio callback runs in its own thread and pushes every block onto a
    queue, so the plot sees each block exactly once instead of sampling
    whatever block happens to be newest.
    """

    def __init__(self, device, fs, block_size, max_queued=64):
        sd = _sounddevice()

        self.q = queue.Queue(maxsize=max_queued)
        self.dropped = 0
        self.stream = sd.InputStream(
            device=device,
            channels=2,
            samplerate=fs,
            blocksize=block_size,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            # Plotting can't keep up; drop the oldest block to stay current.
            self.dropped += 1
            try:
                self.q.get_nowait()
                self.q.put_nowait(indata.copy())
            except (queue.Empty, queue.Full):
                pass

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def poll(self):
        blocks = []
        while True:
            try:
                blocks.append(self.q.get_nowait())
            except queue.Empty:
                return blocks


class FileSource:
    """Stereo blocks from a .wav file, paced at real time and looped."""

    def __init__(self, data, fs, block_size):
        self.data = data
        self.fs = fs
        self.block_size = block_size
        self.n_blocks = len(data) // block_size
        if self.n_blocks == 0:
            raise ValueError("file is shorter than one block; lower --resolution")
        self.emitted = 0
        self.t0 = None

    @classmethod
    def from_wav(cls, path, block_size_for_fs):
        from scipy.io import wavfile

        fs, data = wavfile.read(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{path} must have 2 channels (Q left, I right)")
        data = data[:, :2]
        if data.dtype == np.uint8:
            data = (data.astype(np.float64) - 128) / 128
        elif np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float64) / (np.iinfo(data.dtype).max + 1)
        else:
            data = data.astype(np.float64)
        return cls(data, fs, block_size_for_fs(fs))

    def start(self):
        self.t0 = time.monotonic()

    def stop(self):
        pass

    def poll(self):
        due = int((time.monotonic() - self.t0) * self.fs / self.block_size) + 1
        blocks = []
        while self.emitted < due:
            k = self.emitted % self.n_blocks
            blocks.append(self.data[k * self.block_size:(k + 1) * self.block_size])
            self.emitted += 1
        return blocks


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
class QuickLook:
    def __init__(self, fs, block_size, swap_iq, balance, db_range, history):
        self.fs = fs
        self.n = block_size
        self.swap_iq = swap_iq
        self.balance = balance
        self.window = np.hanning(block_size)
        # Matches MATLAB's -fs/2 : res : fs/2-res for even block sizes
        self.freqs = np.fft.fftshift(np.fft.fftfreq(block_size, d=1.0 / fs))
        self.blocks_seen = 0
        self.zero_blocks = 0     # consecutive all-zero blocks

        self.fig, (self.ax_time, self.ax_spec, self.ax_water) = plt.subplots(
            3, 1, figsize=(10, 9),
            gridspec_kw={"height_ratios": [1, 1, 1.3]},
        )
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("pyquicklook")

        # ---- Time domain: I and Q overlaid ----
        t_ms = np.arange(block_size) / fs * 1e3
        zeros = np.zeros(block_size)
        (self.line_i,) = self.ax_time.plot(t_ms, zeros, label="I (real)", lw=0.8)
        (self.line_q,) = self.ax_time.plot(t_ms, zeros, label="Q (imag)", lw=0.8)
        self.ax_time.set_xlim(0, t_ms[-1])
        self.ax_time.set_ylim(-1, 1)
        self.ax_time.set_xlabel("Time [ms]")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.legend(loc="upper right")
        self.ax_time.grid(True, alpha=0.3)

        # ---- Two-sided spectrum with peak marker ----
        (self.line_spec,) = self.ax_spec.plot(self.freqs, np.full(block_size, DB_FLOOR), lw=0.8)
        (self.peak_dot,) = self.ax_spec.plot([], [], "o", color="C3", ms=5)
        self.peak_text = self.ax_spec.annotate(
            "", xy=(0, 0), xytext=(6, -4), textcoords="offset points",
            va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        self.ax_spec.set_xlim(-fs / 2, fs / 2)
        self.ax_spec.set_ylim(*db_range)
        self.ax_spec.set_xlabel("Frequency [Hz]")
        self.ax_spec.set_ylabel("Magnitude [dBFS]")
        self.ax_spec.grid(True, alpha=0.3)

        # ---- Waterfall: newest row on top, frequency axis shared with spectrum ----
        # Thousands of bins drawn into a few hundred pixels would skip most
        # columns (and the tones in them), so each row keeps the max of every
        # `pool` adjacent bins.
        df = fs / block_size
        self.pool = -(-block_size // WATERFALL_MAX_COLS)
        cols = block_size // self.pool
        self.water = np.full((history, cols), db_range[0])
        left = self.freqs[0] - df / 2
        self.im_water = self.ax_water.imshow(
            self.water,
            aspect="auto",
            origin="upper",
            extent=[left, left + cols * self.pool * df, history, 0],
            cmap="viridis",
            vmin=db_range[0],
            vmax=db_range[1],
            interpolation="nearest",
        )
        self.ax_water.sharex(self.ax_spec)
        self.ax_water.set_xlabel("Frequency [Hz]")
        self.ax_water.set_ylabel("Blocks ago")
        self.fig.colorbar(self.im_water, ax=self.ax_water, label="dBFS", pad=0.01)

        self._set_title(1.0)
        self.fig.tight_layout()

    @property
    def no_input(self):
        """True once the input has been exact digital zeros for a while: a
        real line input always has some noise, so this means the device is
        delivering nothing (e.g. macOS microphone permission not granted)."""
        return self.zero_blocks * self.n / self.fs >= SILENCE_WARN_S

    def _set_title(self, scale):
        bal = f"I gain x{scale:.3f}" if self.balance else "I/Q balance off"
        warn = "\nNO INPUT: the device is returning all zeros" if self.no_input else ""
        self.fig.suptitle(
            f"fs = {self.fs:g} Hz   N = {self.n}   "
            f"resolution = {self.fs / self.n:g} Hz   {bal}{warn}",
            color="C3" if warn else "black",
        )

    def update(self, blocks):
        if not blocks:
            return
        spec = None
        for block in blocks:
            self.zero_blocks = self.zero_blocks + 1 if not np.any(block) else 0
            d, spec, scale = process_block(block, self.window, self.swap_iq, self.balance)
            self.water[1:] = self.water[:-1]
            cols = self.water.shape[1]
            self.water[0] = spec[:cols * self.pool].reshape(cols, self.pool).max(axis=1)
        self.blocks_seen += len(blocks)

        # Only the newest block is drawn in the line plots
        self.line_i.set_ydata(d.real)
        self.line_q.set_ydata(d.imag)
        self.line_spec.set_ydata(spec)
        self.im_water.set_data(self.water)

        k = int(np.argmax(spec))
        f_pk, db_pk = self.freqs[k], spec[k]
        lo, hi = self.ax_spec.get_ylim()
        y = min(max(db_pk, lo), hi)
        self.peak_dot.set_data([f_pk], [y])
        self.peak_text.xy = (f_pk, y)
        self.peak_text.set_text(f"{f_pk:.0f} Hz\n{db_pk:.1f} dBFS")
        # Keep the label on-screen when the peak is near the right edge
        right = f_pk > self.fs / 4
        self.peak_text.set_ha("right" if right else "left")
        self.peak_text.set_position((-6 if right else 6, -4))

        self._set_title(scale)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

GUI_HINT = {
    "Linux": "install Tk ('sudo apt install python3-tk' or your distribution's "
             "equivalent) or Qt ('pip install PyQt6')",
    "Darwin": "use the python.org installer (includes Tk), or with Homebrew "
              "'brew install python-tk', or 'pip install PyQt6'",
    "Windows": "re-run the Python installer with 'tcl/tk and IDLE' ticked, "
               "or 'pip install PyQt6'",
}

SILENCE_HINT = {
    "Darwin": "macOS only delivers audio once the terminal app has microphone "
              "access: System Settings > Privacy & Security > Microphone, then "
              "restart the terminal",
    "Windows": "check Settings > Privacy & security > Microphone > 'Let desktop "
               "apps access your microphone', and that the input isn't muted",
    "Linux": "check the input isn't muted (alsamixer or pavucontrol)",
}


def _hint(table):
    return table.get(platform.system(), table["Linux"])


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Real-time I/Q quicklook (Python port of quicklook_complex.m)"
    )
    p.add_argument("--fs", type=float, default=48000,
                   help="sample rate in Hz (default: 48000; ignored with --file)")
    p.add_argument("--resolution", type=float, default=10,
                   help="FFT bin spacing in Hz; block length is fs/resolution (default: 10)")
    p.add_argument("--device", default=None,
                   help="input device index, or a case-insensitive substring of the "
                        "name or host API as shown by --list (default: system default)")
    p.add_argument("--list", action="store_true",
                   help="list audio input devices and exit")
    p.add_argument("--file", default=None,
                   help="play back a stereo .wav (Q left, I right) instead of live capture")
    p.add_argument("--swap-iq", action="store_true",
                   help="treat left as I and right as Q")
    p.add_argument("--no-iq-balance", action="store_true",
                   help="disable I/Q gain-imbalance correction")
    p.add_argument("--db-range", type=float, nargs=2, default=(-120, 0),
                   metavar=("MIN", "MAX"), help="spectrum dB limits (default: -120 0)")
    p.add_argument("--history", type=int, default=100,
                   help="waterfall depth in blocks (default: 100)")
    p.add_argument("--interval", type=int, default=30,
                   help="plot refresh interval in ms (default: 30)")
    p.add_argument("--save", metavar="IMAGE", default=None,
                   help="run without a window for --duration seconds, then save the "
                        "plots to IMAGE (.png, .pdf, .svg) and exit")
    p.add_argument("--duration", type=float, default=3.0,
                   help="seconds to capture before saving with --save (default: 3)")
    return p.parse_args(argv)


def _block_size(fs, resolution):
    n = int(round(fs / resolution))
    if n < 2:
        raise ValueError("--resolution is too coarse for this sample rate")
    return n


def _input_devices(sd):
    """[(index, device dict, host API name)] for every device with inputs."""
    apis = sd.query_hostapis()
    return [(i, dev, apis[dev["hostapi"]]["name"])
            for i, dev in enumerate(sd.query_devices())
            if dev["max_input_channels"] > 0]


def _select_device(device_arg, fs):
    """Resolve *device_arg* to a sounddevice index that supports 2-channel
    input at *fs*.

    A substring is matched against "name [host API]", so on Windows, where
    each device appears once per host API (MME, DirectSound, WASAPI, WDM-KS),
    "--device WASAPI" or "--device 'X4 [Windows WASAPI]'" can pick one.
    Matches that can't do stereo at *fs* are skipped.
    """
    sd = _sounddevice()
    if device_arg is None:
        candidates = [None]
    else:
        try:
            candidates = [int(device_arg)]
        except ValueError:
            candidates = [i for i, dev, api in _input_devices(sd)
                          if device_arg.lower() in f"{dev['name']} [{api}]".lower()]
            if not candidates:
                raise ValueError(f"no input device matching '{device_arg}' (see --list)")
    problems = []
    for idx in candidates:
        try:
            sd.check_input_settings(device=idx, samplerate=fs, channels=2)
            return idx
        except Exception as e:                                  # noqa: BLE001
            dev = sd.query_devices(idx, "input")
            problems.append(f"  [{'default' if idx is None else idx}] {dev['name']}: "
                            f"{dev['max_input_channels']} input ch, "
                            f"{dev['default_samplerate']:g} Hz default ({e})")
    raise ValueError(
        f"no matching device can record 2 channels at {fs:g} Hz:\n"
        + "\n".join(problems)
        + "\npyquicklook needs a stereo line input; pick one with --list/--device, "
          "or try --fs with the device's default rate"
    )


def _list_devices():
    sd = _sounddevice()
    print("Audio input devices (* = system default):")
    try:
        default_in = sd.default.device[0]
    except Exception:                                           # noqa: BLE001
        default_in = None
    for i, dev, api in _input_devices(sd):
        mark = "*" if i == default_in else " "
        print(f" {mark}[{i}] {dev['name']} [{api}]  "
              f"{dev['max_input_channels']} ch, {dev['default_samplerate']:g} Hz default")


def main(argv=None):
    args = _parse_args(argv)

    try:
        if args.list:
            _list_devices()
            return 0

        if args.save:
            plt.switch_backend("Agg")
        elif plt.get_backend().lower() in NON_INTERACTIVE:
            raise RuntimeError(
                f"matplotlib has no GUI toolkit to open a window with (backend "
                f"'{plt.get_backend()}'); {_hint(GUI_HINT)}. "
                f"Or use --save to write the plots to an image instead."
            )

        if args.file:
            source = FileSource.from_wav(
                args.file, lambda fs: _block_size(fs, args.resolution)
            )
            fs, n = source.fs, source.block_size
            if fs != args.fs:
                print(f"Using file sample rate {fs} Hz")
        else:
            fs = args.fs
            n = _block_size(fs, args.resolution)
            source = LiveSource(_select_device(args.device, fs), fs, n)
    except Exception as e:                                      # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return 1

    ql = QuickLook(
        fs, n,
        swap_iq=args.swap_iq,
        balance=not args.no_iq_balance,
        db_range=tuple(args.db_range),
        history=args.history,
    )
    warned = []

    def frame(_frame=None):
        ql.update(source.poll())
        if ql.no_input and not warned:
            warned.append(True)
            print(f"Warning: the input is all zeros; {_hint(SILENCE_HINT)}",
                  file=sys.stderr)

    source.start()
    try:
        if args.save:
            t_end = time.monotonic() + args.duration
            while time.monotonic() < t_end:
                time.sleep(args.interval / 1000)
                frame()
            ql.fig.savefig(args.save, dpi=100)
            print(f"Saved {args.save}")
        else:
            ani = animation.FuncAnimation(  # noqa: F841 (must stay referenced)
                ql.fig, frame, interval=args.interval,
                blit=False, cache_frame_data=False,
            )
            plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        if isinstance(source, LiveSource) and source.dropped:
            print(f"Note: {source.dropped} blocks dropped because plotting fell behind",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
