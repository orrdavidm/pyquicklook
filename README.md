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

pyquicklook is a command-line app, so install it with **uv** (recommended) or
**pipx**. Each gives the app its own private Python environment and puts the
`pyquicklook` command on your PATH. A plain `pip install` into a Homebrew or
apt-managed Python stops with an "externally-managed-environment" error; these
tools avoid that. Works on Windows, macOS and Linux.

### 1. Get uv (once)

| | |
|---|---|
| **macOS** | `brew install uv` |
| **Windows** | `winget install --id=astral-sh.uv -e` |
| **Linux** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

Other ways to install it are in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).
Then run `uv tool update-shell` and open a new terminal, so the commands uv
installs are on your PATH.

**Linux only:** the audio library comes from your distribution:
`sudo apt install libportaudio2` (Fedora: `sudo dnf install portaudio`). On
macOS and Windows it's included.

### 2. Install pyquicklook

```bash
uv tool install --managed-python git+https://github.com/orrdavidm/pyquicklook
```

`--managed-python` makes uv use its own copy of Python (downloaded the first
time). That copy includes Tk, the toolkit the plot window needs, so nothing
else has to be installed for the window. Installing from GitHub needs `git`;
add `@v0.2.0` (a release tag) to the end of the URL for a particular version.

Without `git`, install a release wheel instead, by URL from the
[Releases](https://github.com/orrdavidm/pyquicklook/releases) page or from a
downloaded `.whl` file:

```bash
uv tool install --managed-python https://github.com/orrdavidm/pyquicklook/releases/download/v0.2.0/pyquicklook-0.2.0-py3-none-any.whl
```

### Updating and removing

```bash
uv tool upgrade pyquicklook      # a GitHub install: fetches the latest code
uv tool uninstall pyquicklook
```

For a wheel install, run the install command again with the new wheel and
`--reinstall`.

### Using pipx instead

Get pipx: `brew install pipx` (macOS), `sudo apt install pipx`
(Ubuntu/Debian) or `py -m pip install --user pipx` (Windows). Then run
`pipx ensurepath` (Windows: `py -m pipx ensurepath`) and open a new terminal.

```bash
pipx install git+https://github.com/orrdavidm/pyquicklook
pipx reinstall pyquicklook       # update: fetches the latest code
pipx uninstall pyquicklook
```

Use `pipx reinstall` to update: `pipx upgrade` only acts when the version
number changes, so it skips new commits. pipx runs the app with the Python
pipx itself uses, so the plot window needs Tk for that Python:
`brew install python-tk` (Homebrew), `sudo apt install python3-tk` (Ubuntu);
the python.org installers include it.

### Qt instead of Tk

If you'd rather use Qt for the window:
`uv tool install --managed-python --with PyQt6 git+https://github.com/orrdavidm/pyquicklook`,
or with pipx, `pipx install "pyquicklook[qt] @ git+https://github.com/orrdavidm/pyquicklook"`.

When the audio library or the window toolkit is missing, pyquicklook says
which one and how to fix it.

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

## Building and sharing a wheel

A wheel (`.whl`) is a ready-to-install package file. To make one for a
release:

1. Bump `version` in `setup.cfg`, so pip can tell the new release from the
   old one.
2. Build:

   ```bash
   uv build
   ```

   This writes `dist/pyquicklook-<version>-py3-none-any.whl`, plus a source
   archive (`.tar.gz`). `dist/` is git-ignored.
3. Share it. The simplest way is a GitHub release, which tags the commit and
   attaches the files:

   ```bash
   gh release create v0.2.0 dist/* --title "pyquicklook 0.2.0" --notes "What changed"
   ```

   (Or on github.com: Releases → Draft a new release, then drag in the files.)
   Students can then install the wheel by URL:

   ```bash
   uv tool install --managed-python https://github.com/orrdavidm/pyquicklook/releases/download/v0.2.0/pyquicklook-0.2.0-py3-none-any.whl
   ```

   You can also post the `.whl` file on Canvas; students install it the same
   way with the file name in place of the URL.

## Working on the code

From a clone of this repository, install your working copy as the
`pyquicklook` command (edits take effect without reinstalling) and run the
tests:

```bash
uv tool install --managed-python --editable .
uv run --no-project --with-editable . --with pytest pytest tests
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
