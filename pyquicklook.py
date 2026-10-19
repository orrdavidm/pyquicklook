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

A one-channel device (e.g. a laptop microphone) or mono .wav also works.  A
real signal has the same content at +f and -f, so it gets a one-sided
(0 .. fs/2) spectrum instead, and a warning says so.

Processing per block mirrors the MATLAB script: remove DC, correct I/Q gain
imbalance, Hann window, FFT, fftshift.  Block length is fs / resolution, so
``--resolution`` is the FFT bin spacing just like MATLAB's
``frequency_resolution`` argument.

Improvements over the MATLAB version:
  * dBFS scaling (a full-scale complex tone reads 0 dB regardless of FFT size)
  * I/Q balance uses RMS rather than mean absolute value
  * waterfall display, device selection, --swap-iq, and --file playback of a
    recorded .wav so the tool can be used without a sound card

The time plot has scope-style controls: a span slider to zoom in (down to a
few samples), and a trigger (rising/falling edge on I or Q at a set level) so
successive captures line up.  Pause (button or space bar) freezes everything,
so the figure's own zoom and pan tools can be used for a closer look.

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
from matplotlib.widgets import Button, Slider

DB_FLOOR = -140.0  # dBFS value used in place of log(0)
WATERFALL_MAX_COLS = 512   # wider spectra are max-pooled so narrow tones stay visible
SILENCE_WARN_S = 2.0       # this long of exact digital zeros means no input at all
MIN_SPAN = 16              # narrowest time-plot span, in samples
PRETRIGGER = 0.1           # fraction of the time-plot span shown before the trigger
TRIGGER_HYSTERESIS = 0.05  # of the signal's peak-to-peak: noise rejection


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


def mono_as_stereo(x):
    """A one-channel signal as a (frames, 2) block with Q = 0 and I = the
    signal, so mono sources fit the same block format.  Process these with
    ``mono=True``."""
    x = np.asarray(x).reshape(len(x), -1)[:, 0]
    return np.column_stack([np.zeros_like(x), x])


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


def spectrum_real_dbfs(x, window):
    """One-sided (0 .. fs/2) magnitude spectrum of a real signal in dBFS.

    The negative-frequency half is the mirror image, so its energy is folded
    in (x2): a full-scale sine reads 0 dBFS, as a complex tone does in
    ``spectrum_dbfs``.  DC and Nyquist have no mirror and aren't doubled.
    """
    mag = 2 * np.abs(np.fft.rfft(x * window)) / np.sum(window)
    mag[0] /= 2
    if len(x) % 2 == 0:
        mag[-1] /= 2
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(mag)
    return np.maximum(db, DB_FLOOR)


def process_block(stereo, window, swap_iq=False, balance=True, mono=False):
    """Run one block through the MATLAB processing chain.

    Returns ``(time_signal, spectrum_db, i_scale)``.  As in MATLAB, the time
    signal is shown after DC removal but *before* I/Q balancing, so any
    imbalance is still visible in the time plot.

    With ``mono=True`` the block comes from ``mono_as_stereo``: the real
    signal in the I column gets a one-sided spectrum, and I/Q balance and
    ``swap_iq`` don't apply.
    """
    if mono:
        x = remove_dc(stereo[:, 1].astype(np.float64))
        return x + 0j, spectrum_real_dbfs(x, window), 1.0
    d = remove_dc(to_complex(stereo, swap_iq))
    balanced, scale = iq_balance(d) if balance else (d, 1.0)
    return d, spectrum_dbfs(balanced, window), scale


def find_trigger(x, level, edge="rise", start=0, stop=None, hysteresis=None):
    """Where *x* first crosses *level* on a rising (or ``edge="fall"``)
    edge, as a fractional sample index, or None.

    Like a scope trigger with hysteresis: for a rising edge the signal must
    first drop below ``level - hysteresis`` before a crossing counts, so noise
    riding on a slow edge doesn't trigger again and again.  The default
    hysteresis is ``TRIGGER_HYSTERESIS`` of the signal's peak-to-peak.  The
    crossing is interpolated between samples so a steady waveform lines up
    exactly from one capture to the next.  Only crossings at indices in
    ``[start, stop)`` count.
    """
    x = np.asarray(x, dtype=np.float64)
    if edge == "fall":
        x, level = -x, -level
    if hysteresis is None:
        hysteresis = TRIGGER_HYSTERESIS * np.ptp(x)
    # -1 = armed (below the band), +1 = at/above the level, 0 = in between;
    # the in-between samples keep whichever state came before them
    state = np.zeros(len(x), dtype=np.int8)
    state[x < level - hysteresis] = -1
    state[x >= level] = 1
    last = np.where(state != 0, np.arange(len(x)), 0)
    np.maximum.accumulate(last, out=last)
    held = state[last]
    k = np.flatnonzero((held[:-1] == -1) & (held[1:] == 1)) + 1
    k = k[(k >= max(start, 1)) & (k < (len(x) if stop is None else stop))]
    if not len(k):
        return None
    k = int(k[0])
    a, b = x[k - 1], x[k]            # a < level <= b
    return k - 1 + (level - a) / (b - a)


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
    """Stereo blocks from a sound card via sounddevice.  A one-channel
    device (``channels=1``) is delivered as I with Q = 0.

    The audio callback runs in its own thread and pushes every block onto a
    queue, so the plot sees each block exactly once instead of sampling
    whatever block happens to be newest.
    """

    def __init__(self, device, fs, block_size, channels=2, max_queued=64):
        sd = _sounddevice()

        self.fs = fs
        self.block_size = block_size
        self.q = queue.Queue(maxsize=max_queued)
        self.dropped = 0
        self.mono = channels == 1
        self.stream = sd.InputStream(
            device=device,
            channels=channels,
            samplerate=fs,
            blocksize=block_size,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        block = mono_as_stereo(indata) if self.mono else indata.copy()
        try:
            self.q.put_nowait(block)
        except queue.Full:
            # Plotting can't keep up; drop the oldest block to stay current.
            self.dropped += 1
            try:
                self.q.get_nowait()
                self.q.put_nowait(block)
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
    """Stereo blocks from a .wav file, paced at real time and looped.  A mono
    file is delivered as I with Q = 0."""

    def __init__(self, data, fs, block_size, mono=False):
        self.data = data
        self.mono = mono
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
        mono = data.ndim == 1 or data.shape[1] == 1
        data = mono_as_stereo(data) if mono else data[:, :2]
        if data.dtype == np.uint8:
            data = (data.astype(np.float64) - 128) / 128
        elif np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float64) / (np.iinfo(data.dtype).max + 1)
        else:
            data = data.astype(np.float64)
        return cls(data, fs, block_size_for_fs(fs), mono)

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
    """The three-panel figure: time domain (with span, trigger and pause
    controls), spectrum with peak marker, and waterfall.  Feed it blocks from
    a source with ``update()``."""

    TRIGGER_MODES = ("off", "rise", "fall")
    TRIGGER_LABELS = {"off": "Trigger: off", "rise": "Trigger: rise",
                      "fall": "Trigger: fall"}

    def __init__(self, fs, block_size, swap_iq, balance, db_range, history, mono=False,
                 span_ms=None, trigger="off", trigger_level=0.0, trigger_source="I"):
        self.fs = fs
        self.n = block_size
        self.swap_iq = swap_iq
        self.balance = balance
        self.mono = mono
        self.window = np.hanning(block_size)
        if mono:
            # A real signal: 0 .. fs/2 only
            self.freqs = np.fft.rfftfreq(block_size, d=1.0 / fs)
        else:
            # Matches MATLAB's -fs/2 : res : fs/2-res for even block sizes
            self.freqs = np.fft.fftshift(np.fft.fftfreq(block_size, d=1.0 / fs))
        self.blocks_seen = 0
        self.zero_blocks = 0     # consecutive all-zero blocks
        self.paused = False
        self.last_d = np.zeros(block_size, dtype=complex)
        self.trigger = trigger
        self.trigger_level = trigger_level
        self.trigger_source = "I" if mono else trigger_source
        self.triggered = None    # None (trigger off), True, or False (no edge found)

        self.fig = plt.figure(figsize=(10, 10.5))
        grid = self.fig.add_gridspec(4, 1, height_ratios=[1.15, 0.22, 1, 1.25],
                                     left=0.08, right=0.97, top=0.93, bottom=0.05,
                                     hspace=0.4)
        self.ax_time = self.fig.add_subplot(grid[0])
        self.ax_spec = self.fig.add_subplot(grid[2])
        self.ax_water = self.fig.add_subplot(grid[3])
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("pyquicklook")
        self._setup_time_plot()
        self._setup_controls(grid[1], span_ms)
        self._setup_spectrum(db_range)
        self._setup_waterfall(db_range, history)
        self._set_title(1.0)
        self._draw_time(self.last_d)

    # ---- figure setup ------------------------------------------------------
    def _setup_time_plot(self):
        """I and Q overlaid (just the input for mono)."""
        ax = self.ax_time
        t_ms = np.arange(self.n) / self.fs * 1e3
        zeros = np.zeros(self.n)
        (self.line_i,) = ax.plot(
            t_ms, zeros, label="mono input" if self.mono else "I (real)", lw=0.8)
        (self.line_q,) = ax.plot(t_ms, zeros, label="Q (imag)", lw=0.8)
        if self.mono:
            self.line_q.remove()
        # trigger level and trigger instant (t = 0) markers
        self.trig_hline = ax.axhline(0, color="C3", lw=0.8, ls="--", alpha=0.7)
        self.trig_vline = ax.axvline(0, color="C3", lw=0.8, ls=":", alpha=0.7)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _setup_controls(self, cell, span_ms):
        """Two rows under the time plot:

            Time span      [slider]  readout  [Reset]          [Pause]
            Trigger level  [slider]  readout  [Reset]  [Trigger: off] [Src: I]

        Reset puts a slider back where it started (the command-line value, or
        the whole block / 0 by default)."""
        # column 0 holds each slider's name, column 2 its readout
        rows = cell.subgridspec(2, 7, width_ratios=[1.05, 4.2, 1.25, 0.75, 0.25, 1.25, 0.8],
                                wspace=0.08, hspace=0.6)
        fig = self.fig

        # Time span: a log slider from MIN_SPAN samples to the whole block
        self.span_lo = min(MIN_SPAN, self.n)
        self.span_ax = fig.add_subplot(rows[0, 1])
        self.span_slider = Slider(self.span_ax, "Time span", 0.0, 1.0, valinit=1.0)
        if span_ms is not None:
            self.set_span_ms(span_ms)
        self.span_reset_val = self.span_slider.val
        self.span_slider.on_changed(lambda _v: self._controls_changed())
        self.span_reset = Button(fig.add_subplot(rows[0, 3]), "Reset")
        self.span_reset.on_clicked(lambda _e: self.reset_span())
        self.pause_button = Button(fig.add_subplot(rows[0, 5:]), "Pause")
        self.pause_button.on_clicked(lambda _e: self.toggle_pause())

        # Trigger level, mode and source
        self.level_ax = fig.add_subplot(rows[1, 1])
        self.level_reset_val = float(np.clip(self.trigger_level, -1, 1))
        self.level_slider = Slider(self.level_ax, "Trigger level", -1.0, 1.0,
                                   valinit=self.level_reset_val, valfmt="%+.2f")
        self.level_slider.on_changed(self._level_changed)
        self.level_reset = Button(fig.add_subplot(rows[1, 3]), "Reset")
        self.level_reset.on_clicked(lambda _e: self.reset_level())
        self.trig_button = Button(fig.add_subplot(rows[1, 5]), self.TRIGGER_LABELS[self.trigger])
        self.trig_button.on_clicked(self._cycle_trigger)
        self.src_ax = fig.add_subplot(rows[1, 6])
        self.src_button = Button(self.src_ax, f"Src: {self.trigger_source}")
        self.src_button.on_clicked(self._toggle_source)
        if self.mono:
            self.src_ax.set_visible(False)

        for slider in (self.span_slider, self.level_slider):
            slider.label.set_fontsize(10)
        fig.canvas.mpl_connect("key_press_event", self._key)
        self._update_span_text()

    def _setup_spectrum(self, db_range):
        """Two-sided spectrum (one-sided for mono) with a peak marker."""
        ax = self.ax_spec
        (self.line_spec,) = ax.plot(self.freqs, np.full(len(self.freqs), DB_FLOOR), lw=0.8)
        (self.peak_dot,) = ax.plot([], [], "o", color="C3", ms=5)
        self.peak_text = ax.annotate(
            "", xy=(0, 0), xytext=(6, -4), textcoords="offset points",
            va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax.set_xlim(0 if self.mono else -self.fs / 2, self.fs / 2)
        ax.set_ylim(*db_range)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dBFS]")
        ax.grid(True, alpha=0.3)

    def _setup_waterfall(self, db_range, history):
        """Newest row on top, frequency axis shared with the spectrum.

        Thousands of bins drawn into a few hundred pixels would skip most
        columns (and the tones in them), so each row keeps the max of every
        ``pool`` adjacent bins.
        """
        ax = self.ax_water
        nbins = len(self.freqs)
        df = self.fs / self.n
        self.pool = -(-nbins // WATERFALL_MAX_COLS)
        cols = nbins // self.pool
        self.water = np.full((history, cols), db_range[0])
        left = self.freqs[0] - df / 2
        self.im_water = ax.imshow(
            self.water,
            aspect="auto",
            origin="upper",
            extent=[left, left + cols * self.pool * df, history, 0],
            cmap="viridis",
            vmin=db_range[0],
            vmax=db_range[1],
            interpolation="nearest",
        )
        ax.sharex(self.ax_spec)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Blocks ago")
        self.fig.colorbar(self.im_water, ax=ax, label="dBFS", pad=0.01)

    # ---- time-plot controls ------------------------------------------------
    @property
    def span_samples(self):
        """Time-plot span from the log slider: MIN_SPAN .. block length."""
        lo, hi = self.span_lo, self.n
        return int(round(lo * (hi / lo) ** self.span_slider.val))

    def set_span_ms(self, ms):
        lo, hi = self.span_lo, self.n
        samples = min(max(ms * 1e-3 * self.fs, lo), hi)
        self.span_slider.set_val(np.log(samples / lo) / np.log(hi / lo) if hi > lo else 1.0)

    def reset_span(self):
        """Back to the starting span (--span, or the whole block)."""
        self.span_slider.set_val(self.span_reset_val)

    def reset_level(self):
        """Back to the starting trigger level (--trigger-level, or 0)."""
        self.level_slider.set_val(self.level_reset_val)

    def _update_span_text(self):
        ms = self.span_samples / self.fs * 1e3
        self.span_slider.valtext.set_text(
            f"{ms:.3g} ms ({self.span_samples})" if ms < 10 else f"{ms:.0f} ms")

    def _level_changed(self, value):
        self.trigger_level = value
        self._controls_changed()

    def _cycle_trigger(self, _event=None):
        modes = self.TRIGGER_MODES
        self.trigger = modes[(modes.index(self.trigger) + 1) % len(modes)]
        self.trig_button.label.set_text(self.TRIGGER_LABELS[self.trigger])
        self._controls_changed()

    def _toggle_source(self, _event=None):
        if not self.mono:
            self.trigger_source = "Q" if self.trigger_source == "I" else "I"
            self.src_button.label.set_text(f"Src: {self.trigger_source}")
            self._controls_changed()

    def toggle_pause(self):
        """Freeze or resume the whole display.  While frozen, the figure's
        zoom and pan tools work on every plot, as after stopping the MATLAB
        script."""
        self.paused = not self.paused
        self.pause_button.label.set_text("Run" if self.paused else "Pause")
        self._set_title(self.last_scale)
        self.fig.canvas.draw_idle()

    def _key(self, event):
        if event.key == " ":
            self.toggle_pause()

    def _controls_changed(self):
        """Redraw the time plot from the newest block, so the controls work
        while paused too."""
        self._update_span_text()
        self._draw_time(self.last_d)
        self.fig.canvas.draw_idle()

    def _draw_time(self, d):
        """Plot *d* with t = 0 at the trigger point (if any), over the span."""
        span = self.span_samples
        pre = int(round(PRETRIGGER * span))
        t0 = 0.0
        self.triggered = None
        if self.trigger != "off":
            sig = d.real if self.trigger_source == "I" else d.imag
            # Prefer a trigger that leaves the whole span inside the block; for
            # wide spans accept one in the first half (the right end of the
            # plot is then empty, as on a scope)
            k = find_trigger(sig, self.trigger_level, self.trigger, start=pre,
                             stop=max(self.n - (span - pre) + 1, self.n // 2))
            self.triggered = k is not None
            if k is not None:
                t0 = k / self.fs
        t_ms = (np.arange(self.n) / self.fs - t0) * 1e3
        self.line_i.set_data(t_ms, d.real)
        if not self.mono:
            self.line_q.set_data(t_ms, d.imag)
        span_ms = span / self.fs * 1e3
        lo = -pre / self.fs * 1e3 if self.triggered else 0.0
        self.ax_time.set_xlim(lo, lo + span_ms)

        on = self.trigger != "off"
        self.trig_hline.set_visible(on)
        self.trig_hline.set_ydata([self.trigger_level] * 2)
        self.trig_vline.set_visible(bool(self.triggered))
        if not on:
            state = "free running"
        elif self.triggered:
            state = (f"triggered on {self.trigger_source} "
                     f"{'rising' if self.trigger == 'rise' else 'falling'} "
                     f"through {self.trigger_level:+.2f}")
        else:
            state = "no trigger edge in this block: free running"
        self.ax_time.set_title(state, fontsize=9, loc="left",
                               color="black" if self.triggered is not False else "C3")

    # ---- per-frame updates -------------------------------------------------
    @property
    def no_input(self):
        """True once the input has been exact digital zeros for a while: a
        real line input always has some noise, so this means the device is
        delivering nothing (e.g. macOS microphone permission not granted)."""
        return self.zero_blocks * self.n / self.fs >= SILENCE_WARN_S

    def update(self, blocks):
        """Process every new block into the waterfall, then redraw the line
        plots, peak marker and title from the newest one."""
        if not blocks or self.paused:
            return
        for block in blocks:
            d, spec, scale = self._process(block)
        self.blocks_seen += len(blocks)
        self.last_d = d

        self._draw_time(d)
        self.line_spec.set_ydata(spec)
        self.im_water.set_data(self.water)
        self._mark_peak(spec)
        self._set_title(scale)

    def _process(self, block):
        """One block: track silence, compute it, scroll it into the waterfall."""
        self.zero_blocks = self.zero_blocks + 1 if not np.any(block) else 0
        d, spec, scale = process_block(block, self.window, self.swap_iq,
                                       self.balance, self.mono)
        cols = self.water.shape[1]
        self.water[1:] = self.water[:-1]
        self.water[0] = spec[:cols * self.pool].reshape(cols, self.pool).max(axis=1)
        return d, spec, scale

    def _mark_peak(self, spec):
        """Dot and "Hz / dBFS" label on the strongest bin."""
        k = int(np.argmax(spec))
        f_pk, db_pk = self.freqs[k], spec[k]
        lo, hi = self.ax_spec.get_ylim()
        y = min(max(db_pk, lo), hi)
        self.peak_dot.set_data([f_pk], [y])
        self.peak_text.xy = (f_pk, y)
        self.peak_text.set_text(f"{f_pk:.0f} Hz\n{db_pk:.1f} dBFS")
        # Keep the label on-screen when the peak is near the right edge
        x0, x1 = self.ax_spec.get_xlim()
        right = f_pk > x1 - (x1 - x0) / 4
        self.peak_text.set_ha("right" if right else "left")
        self.peak_text.set_position((-6 if right else 6, -4))

    def _set_title(self, scale):
        self.last_scale = scale
        if self.mono:
            bal = "MONO INPUT: one-sided spectrum"
        elif self.balance:
            bal = f"I gain x{scale:.3f}"
        else:
            bal = "I/Q balance off"
        warn = "\nNO INPUT: the device is returning all zeros" if self.no_input else ""
        paused = ("\nPAUSED: zoom/pan with the toolbar; Run or space bar to resume"
                  if self.paused else "")
        self.fig.suptitle(
            f"fs = {self.fs:g} Hz   N = {self.n}   "
            f"resolution = {self.fs / self.n:g} Hz   {bal}{warn}{paused}",
            color="C3" if warn or paused else "black",
        )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

UV_TK_FIX = ("the simplest fix is to reinstall with uv's own Python, which "
             "includes Tk: 'uv tool install --reinstall --managed-python "
             "git+https://github.com/orrdavidm/pyquicklook'")

GUI_HINT = {
    "Linux": UV_TK_FIX + "; or install Tk for this Python "
             "('sudo apt install python3-tk' or your distribution's equivalent)",
    "Darwin": UV_TK_FIX + "; or install Tk for this Python "
              "(Homebrew: 'brew install python-tk')",
    "Windows": UV_TK_FIX + "; or re-run the Python installer with 'tcl/tk and "
               "IDLE' ticked",
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
                   help="play back a .wav (stereo: Q left, I right; mono gets a "
                        "one-sided spectrum) instead of live capture")
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
    p.add_argument("--span", type=float, default=None, metavar="MS",
                   help="time-plot span in ms (default: the whole block); also a "
                        "slider in the window")
    p.add_argument("--trigger", choices=("off", "rise", "fall"), default="off",
                   help="time-plot trigger edge (default: off); also a button")
    p.add_argument("--trigger-level", type=float, default=0.0, metavar="LEVEL",
                   help="trigger level, -1 .. 1 (default: 0); also a slider")
    p.add_argument("--trigger-source", choices=("I", "Q"), default="I",
                   help="channel the trigger watches (default: I); also a button")
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
    """Resolve *device_arg* to ``(sounddevice index, channels)`` for input at
    *fs*.

    A substring is matched against "name [host API]", so on Windows, where
    each device appears once per host API (MME, DirectSound, WASAPI, WDM-KS),
    "--device WASAPI" or "--device 'X4 [Windows WASAPI]'" can pick one.
    Matches that can't record at *fs* are skipped.  Any match that can record
    stereo wins; otherwise a one-channel device (e.g. a laptop microphone) is
    used with ``channels=1``.
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
    problems = {}
    for channels in (2, 1):
        for idx in candidates:
            try:
                sd.check_input_settings(device=idx, samplerate=fs, channels=channels)
                return idx, channels
            except Exception as e:                              # noqa: BLE001
                problems.setdefault(idx, e)
    lines = []
    for idx, e in problems.items():
        dev = sd.query_devices(idx, "input")
        lines.append(f"  [{'default' if idx is None else idx}] {dev['name']}: "
                     f"{dev['max_input_channels']} input ch, "
                     f"{dev['default_samplerate']:g} Hz default ({e})")
    raise ValueError(
        f"no matching device can record at {fs:g} Hz:\n" + "\n".join(lines)
        + "\npick another with --list/--device, or try --fs with the device's "
          "default rate"
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


def _check_gui(save):
    """Use the windowless backend for --save; otherwise make sure matplotlib
    found a GUI toolkit to open a window with."""
    if save:
        plt.switch_backend("Agg")
    elif plt.get_backend().lower() in NON_INTERACTIVE:
        raise RuntimeError(
            f"matplotlib has no GUI toolkit to open a window with (backend "
            f"'{plt.get_backend()}'); {_hint(GUI_HINT)}. "
            f"Or use --save to write the plots to an image instead."
        )


def _open_source(args):
    """The --file or sound-card source, with a warning if it is mono."""
    if args.file:
        source = FileSource.from_wav(args.file, lambda fs: _block_size(fs, args.resolution))
        if source.fs != args.fs:
            print(f"Using file sample rate {source.fs} Hz")
        what = f"{args.file} is mono"
    else:
        n = _block_size(args.fs, args.resolution)
        device, channels = _select_device(args.device, args.fs)
        source = LiveSource(device, args.fs, n, channels)
        what = "the input device has only one channel"
    if source.mono:
        print(f"Warning: {what}; showing a one-sided spectrum (0 to fs/2). One "
              f"channel can't tell +f from -f, so I/Q balance and --swap-iq "
              f"don't apply. Use a stereo line input for real I/Q.",
              file=sys.stderr)
    return source


def _run(ql, source, args):
    """Feed the figure until the window closes, or for --duration with --save."""
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


def main(argv=None):
    args = _parse_args(argv)
    try:
        if args.list:
            _list_devices()
            return 0
        _check_gui(args.save)
        source = _open_source(args)
    except Exception as e:                                      # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return 1

    ql = QuickLook(
        source.fs, source.block_size,
        swap_iq=args.swap_iq,
        balance=not args.no_iq_balance,
        db_range=tuple(args.db_range),
        history=args.history,
        mono=source.mono,
        span_ms=args.span,
        trigger=args.trigger,
        trigger_level=args.trigger_level,
        trigger_source=args.trigger_source,
    )
    _run(ql, source, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
