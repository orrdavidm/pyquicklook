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
add `@v0.2.1` (a release tag) to the end of the URL for a particular version.

Without `git`, install a release wheel instead, by URL from the
[Releases](https://github.com/orrdavidm/pyquicklook/releases) page or from a
downloaded `.whl` file:

```bash
uv tool install --managed-python https://github.com/orrdavidm/pyquicklook/releases/download/v0.2.1/pyquicklook-0.2.1-py3-none-any.whl
```

### Updating

To see which version you have, run `uv tool list` (or `pipx list --short`).
The newest release is at the top of the
[Releases](https://github.com/orrdavidm/pyquicklook/releases) page.

How to update depends on how you installed:

| You installed from | To update |
|---|---|
| GitHub, with no `@` in the URL | `uv tool upgrade pyquicklook` (fetches the latest code from `main`) |
| a release wheel, or GitHub with `@` a tag or branch | run the install command again with the new release and `--reinstall` (below) |

`uv tool upgrade` only follows what you originally installed. A wheel or an
`@v0.2.0` install stays where it is ("Nothing to upgrade"), and an `@branch`
install fails with "Failed to upgrade pyquicklook" once that branch is
deleted. Installing again moves you to whatever you give it, whatever you had
before. For example, to the 0.2.1 release:

```bash
uv tool install --reinstall --managed-python https://github.com/orrdavidm/pyquicklook/releases/download/v0.2.1/pyquicklook-0.2.1-py3-none-any.whl
```

or to the latest code (after which `uv tool upgrade` keeps it current):

```bash
uv tool install --reinstall --managed-python git+https://github.com/orrdavidm/pyquicklook
```

To remove pyquicklook: `uv tool uninstall pyquicklook`.

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

### Looking closely at the time plot

The two rows of controls under the time plot work like a scope's front panel:

* **Time span** zooms the time axis, from the whole block down to 16 samples
  (a log slider), e.g. to look at single cycles of a sine wave for
  discontinuities.
* **Trigger level** sets the level, the **Trigger** button cycles
  off / rise / fall, and **Src** picks the channel it watches (I or Q). Each
  capture is then drawn with t = 0 at the first crossing (interpolated
  between samples, with a little hysteresis so noise doesn't re-trigger), so
  successive captures line up and a steady waveform stands still. If no edge
  is found in a block, the plot free-runs and says so, like a scope in Auto
  mode.
* **Type a value** in the box next to a slider and press Enter to set it
  exactly. Time span is in ms unless you give a unit: `2`, `500 us`,
  `0.01 s` or `100 samples`. Trigger level is −1 to 1. Anything else is
  ignored and the box shows the current value again.
* **Reset** next to each slider puts it back where it started: the whole
  block and level 0, or the values given on the command line.
* **Pause** (or the space bar) freezes every plot. While paused, the
  figure toolbar's zoom and pan tools work on any of the plots, the way
  stopping `quicklook_complex.m` used to; press **Run** or space to continue.

The same settings can be given on the command line, e.g. for `--save`:

```bash
pyquicklook --span 2 --trigger rise --trigger-level 0.1 --trigger-source I
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
(waterfall depth in blocks), `--interval MS` (shortest time between screen
refreshes; pyquicklook refreshes less often if drawing is slow, so the window
stays responsive) and
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
   gh release create v0.2.1 dist/* --title "pyquicklook 0.2.1" --notes "What changed"
   ```

   (Or on github.com: Releases → Draft a new release, then drag in the files.)
   Students can then install the wheel by URL:

   ```bash
   uv tool install --managed-python https://github.com/orrdavidm/pyquicklook/releases/download/v0.2.1/pyquicklook-0.2.1-py3-none-any.whl
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
