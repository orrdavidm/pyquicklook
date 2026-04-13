# audio_visualizer.py
"""
A simple Python program to capture audio from the default input device and
visualize:
  * Time‑domain waveform (amplitude vs. time)
  * Frequency spectrum (FFT magnitude)
  * Waterfall / spectrogram (frequency vs. time)

Dependencies
------------
- numpy
- matplotlib
- sounddevice (for real‑time audio capture)
- scipy (optional, for window functions)

Install them with::
    pip install numpy matplotlib sounddevice scipy

The script uses matplotlib's animation API to update three sub‑plots in
real‑time. It works on Linux, macOS and Windows as long as a working audio
backend is available.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sounddevice as sd
from scipy.signal import get_window
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100          # Sampling frequency (Hz)
BLOCK_SIZE = 1024           # Number of samples per audio block
FFT_SIZE = 1024             # FFT size – typically same as BLOCK_SIZE
WINDOW = "hann"            # Window type for spectrum / spectrogram
UPDATE_INTERVAL = 30       # Plot update interval in milliseconds

# ---------------------------------------------------------------------------
# Command‑line arguments (optional device selection)
# ---------------------------------------------------------------------------
def _parse_args():
    """Parse optional command‑line arguments.

    ``--device`` can be either an integer index or a substring of the device name.
    ``--list`` prints the available input devices and exits.
    """
    parser = argparse.ArgumentParser(description="Real‑time audio visualizer")
    parser.add_argument(
        "--device",
        help="Audio input device index or name substring (default: system default)",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--list",
        help="List available audio input devices and exit",
        action="store_true",
    )
    return parser.parse_args()

def _select_device(device_arg: str | None):
    """Resolve *device_arg* to a sounddevice device index.

    If *device_arg* is ``None`` the default system device is used.
    If it is an integer string it is interpreted directly as the device index.
    Otherwise we search for the first input device whose name contains the
    supplied substring (case‑insensitive).  Raises ``ValueError`` if not found.
    """
    if device_arg is None:
        return None
    # try integer index first
    try:
        idx = int(device_arg)
        sd.check_input_settings(device=idx, samplerate=SAMPLE_RATE, channels=1)
        return idx
    except Exception:
        pass
    # fallback to substring match
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and device_arg.lower() in dev["name"].lower():
            return i
    raise ValueError(f"No input device matching '{device_arg}' found")

# Pre‑compute window coefficients
window = get_window(WINDOW, FFT_SIZE)

# Buffers for waterfall (spectrogram) – we keep a limited number of columns
WATERFALL_HISTORY = 100  # Number of spectra to keep on the waterfall
# NOTE: np.fft.rfft on an N‑point signal returns N/2 + 1 bins (including Nyquist).
# The original code allocated only FFT_SIZE // 2 rows, which caused a broadcasting
# error when assigning the magnitude spectrum (shape 513 for FFT_SIZE=1024).
# Allocate one extra row to match the FFT output size.
spectrogram_data = np.zeros((FFT_SIZE // 2 + 1, WATERFALL_HISTORY))

# ---------------------------------------------------------------------------
# Matplotlib figure setup
# ---------------------------------------------------------------------------
fig, (ax_time, ax_spec, ax_water) = plt.subplots(
    3, 1, figsize=(8, 8), sharex=False
)

# Time‑domain plot (will be updated with the latest block)
line_time, = ax_time.plot(np.arange(BLOCK_SIZE) / SAMPLE_RATE,
                          np.zeros(BLOCK_SIZE))
ax_time.set_title("Time Domain")
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel("Amplitude")
ax_time.set_xlim(0, BLOCK_SIZE / SAMPLE_RATE)
ax_time.set_ylim(-1.0, 1.0)

# Frequency‑domain (single‑sided magnitude spectrum)
freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)
line_spec, = ax_spec.plot(freqs, np.zeros_like(freqs))
ax_spec.set_title("Frequency Spectrum")
ax_spec.set_xlabel("Frequency [Hz]")
ax_spec.set_ylabel("Magnitude (dB)")
ax_spec.set_xlim(0, SAMPLE_RATE / 2)
ax_spec.set_ylim(-120, 0)

# Waterfall / spectrogram image
im_water = ax_water.imshow(
    spectrogram_data,
    aspect="auto",
    origin="lower",
    extent=[0, WATERFALL_HISTORY, 0, SAMPLE_RATE / 2],
    cmap="viridis",
    vmin=-120,
    vmax=0,
)
ax_water.set_title("Spectrogram (Waterfall)")
ax_water.set_xlabel("Time (blocks)")
ax_water.set_ylabel("Frequency [Hz]")
ax_water.set_ylim(0, SAMPLE_RATE / 2)

# ---------------------------------------------------------------------------
# Audio callback – runs in a separate thread managed by sounddevice
# ---------------------------------------------------------------------------

def audio_callback(indata, frames, time, status):
    """Callback receives a block of audio data as a NumPy array.

    * ``indata`` – shape (frames, channels). We use only the first channel.
    * ``frames`` – should match ``BLOCK_SIZE``.
    * ``status`` – reports under‑/over‑flows.
    """
    if status:
        print(status)
    # Ensure mono audio
    audio_block = indata[:, 0]
    # Store the block for the animation routine via a global variable
    global latest_block
    latest_block = audio_block

# Global variable that the animation function will read
latest_block = np.zeros(BLOCK_SIZE)

# ---------------------------------------------------------------------------
# Matplotlib animation function
# ---------------------------------------------------------------------------
def update_plot(frame):
    """Update the three sub‑plots using the most recent audio block.
    ``frame`` is supplied by ``FuncAnimation`` but is not used.
    """
    # Grab the newest audio data (copied to avoid race conditions)
    audio = latest_block.copy()

    # ---- Time domain ----
    line_time.set_ydata(audio)

    # ---- Frequency spectrum ----
    # Apply window, compute FFT, convert to dBFS
    windowed = audio * window
    spectrum = np.fft.rfft(windowed)
    magnitude = 20 * np.log10(np.abs(spectrum) + 1e-12)  # avoid log(0)
    line_spec.set_ydata(magnitude)

    # ---- Spectrogram (waterfall) ----
    spectrogram_data[:, :-1] = spectrogram_data[:, 1:]
    spectrogram_data[:, -1] = magnitude
    im_water.set_data(spectrogram_data)
    im_water.set_extent([0, WATERFALL_HISTORY, 0, SAMPLE_RATE / 2])

    return line_time, line_spec, im_water

def main():
    """Entry point for the command‑line script.
    
    All the logic that was previously under ``if __name__ == "__main__"``
    has been moved into this function so that packaging tools can expose a
    ``console_scripts`` entry point.
    """
    # -------------------------------------------------------------------
    # Parse command‑line arguments and optionally list/select devices
    # -------------------------------------------------------------------
    args = _parse_args()
    if args.list:
        # Print a nicely formatted list of input devices and exit
        print("Available audio input devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"  [{i}] {dev['name']}")
        raise SystemExit(0)

    try:
        selected_device = _select_device(args.device)
    except Exception as e:
        print(f"Error selecting audio device: {e}")
        raise SystemExit(1)

    # Open an input stream – blocksize must match ``BLOCK_SIZE`` for low latency
    # ``device`` is None to use the system default, otherwise the resolved index.
    with sd.InputStream(
        device=selected_device,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        callback=audio_callback,
    ):
        # Start the animation loop; the stream runs in a background thread
        ani = animation.FuncAnimation(
            fig,
            update_plot,
            interval=UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

"""
The program runs until the user closes the Matplotlib window.
If you need to record to a file, you can extend the callback to write the
samples to a ``wav`` file using ``scipy.io.wavfile.write`` or the ``soundfile``
package.
"""
