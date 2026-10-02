# pyquicklook

Real-time I/Q quicklook. This is a Python port of `quicklook_complex.m`.

The left and right channels of a stereo sound-card input are treated as one
complex baseband signal. **Left is Q (imaginary) and right is I (real)**, the
same as the sample bitfile and the ZED playback. The window shows three live
plots:

* **Time domain**: I and Q overlaid
* **Spectrum**: two-sided (−fs/2 … +fs/2) Hann-windowed FFT in dBFS, with the
  strongest bin labelled in Hz and dB
* **Waterfall**: recent spectrum history on the same frequency axis

Each block is processed like the MATLAB script: remove DC, correct I/Q gain
imbalance, apply a window, FFT and `fftshift`. The block length is
`fs / resolution`, so `--resolution` is the FFT bin spacing, the same as
MATLAB's `frequency_resolution` argument.

## Install

Needs Python 3.10 or newer. It works on Windows, macOS and Linux.

```bash
pip install git+https://github.com/orrdavidm/pyquicklook
```

or from a clone of this repository, `pip install .` (add `-e` to work on the
code).

| | Audio (PortAudio) | Plot window (Tk or Qt) |
|---|---|---|
| **Windows** | included in the `sounddevice` wheel | included with the python.org installer ("tcl/tk" option) |
| **macOS** | included in the `sounddevice` wheel | python.org installer: included. Homebrew: `brew install python-tk` |
| **Linux** | `sudo apt install libportaudio2` (Fedora: `portaudio`) | `sudo apt install python3-tk` |

If Tk isn't available, `pip install "pyquicklook[qt] @ git+https://github.com/orrdavidm/pyquicklook"` installs Qt instead.
When one of these pieces is missing, pyquicklook says which one and how to
install it.

**macOS microphone permission.** The first time it records, macOS asks whether
the terminal app may use the microphone. If that was refused, the recording is
all zeros and the plot title shows **NO INPUT**. To fix it, allow the terminal
under System Settings > Privacy & Security > Microphone, then restart it.
Windows has a similar switch: Settings > Privacy & security > Microphone >
"Let desktop apps access your microphone".

## Usage

```bash
# Equivalent to quicklook_complex(10, 48000)
pyquicklook --fs 48000 --resolution 10

# List input devices (with host API), then pick one by index or substring
pyquicklook --list
pyquicklook --device 8
pyquicklook --device "Sound Blaster"

# Replay a recorded stereo capture instead of using the sound card
pyquicklook --file capture.wav

# No window: capture 3 s and save the plots, e.g. for a lab report
pyquicklook --save quicklook.png

# Channels wired the other way round / see the raw imbalance
pyquicklook --swap-iq
pyquicklook --no-iq-balance
```

**Mono inputs.** Real I/Q needs a stereo line input, but a one-channel device
(such as a built-in or USB microphone) or a mono `.wav` still works, which is
handy for trying pyquicklook out. One channel is a real signal, with the same
content at +f and −f, so it gets a **one-sided spectrum** from 0 to fs/2 (a
full-scale sine still reads 0 dBFS). I/Q balance and `--swap-iq` don't apply.
pyquicklook prints a warning and the plot title says **MONO INPUT**. When a
`--device` substring matches both kinds, a stereo device is chosen.

`--device` matches the name *or* the host API shown by `--list`. On Windows
each card is listed once per host API (MME, DirectSound, WASAPI, WDM-KS), so
`--device WASAPI` or `--device "X4 [Windows WASAPI]"` picks one. Matches that
can't record stereo at `--fs` are skipped. WASAPI usually only accepts the
rate set in the Windows Sound control panel; MME and DirectSound accept any
rate.

Other options are `--db-range MIN MAX` (default `-120 0`), `--history N`
(waterfall depth in blocks), `--interval MS` (plot refresh time) and
`--duration S` (capture time for `--save`). Close the window or press Ctrl+C
to stop.

## Tests

```bash
pip install pytest
pytest tests
```

The tests need no sound card and no display. CI runs them on Windows, macOS
and Linux (`.github/workflows/tests.yml`).

## Differences from the MATLAB version

* The spectrum is in **dBFS**. A full-scale complex tone reads 0 dB whatever
  the FFT length, so levels look different from MATLAB's unscaled
  `20*log10(abs(fft))`.
* I/Q balance matches **RMS** levels. MATLAB matches mean absolute value
  (under variables named `rrms`/`irms`). The applied I gain is shown in the
  title bar.
* Every captured block is processed, so the waterfall time axis is accurate.
  If plotting can't keep up, the oldest blocks are dropped and the number
  dropped is printed on exit.

## License

MIT, see [LICENSE](LICENSE).
