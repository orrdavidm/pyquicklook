# pyquicklook

Real‑time audio visualizer that shows three live plots:

* **Time‑domain waveform** – amplitude vs. time
* **Frequency spectrum** – magnitude (dB) of the FFT
* **Spectrogram / waterfall** – frequency vs. time

The program captures audio from any input device, lets you list and select the desired device, and updates the visualizations using Matplotlib's animation API.

## Installation

You can install the package from a wheel, source distribution, or directly from a repository.

```bash
# From a built distribution (wheel or sdist)
pip install pyquicklook‑0.1.0‑py3‑none‑any.whl
```

If you publish to a private PyPI‑compatible index (e.g., Artifactory):

```bash
pip install --index-url https://artifactory.mycompany.com/artifactory/api/pypi/pypi pyquicklook
```

## Usage

```bash
# Use the default audio input device
pyquicklook

# List available input devices
pyquicklook --list

# Select a device by index or by a name substring (case‑insensitive)
pyquicklook --device 2
pyquicklook --device "USB"
```

## Development

If you want to work on the source code locally:

```bash
# Clone the repository (once you push it to GitHub)
git clone https://github.com/yourusername/pyquicklook.git
cd pyquicklook

# Create a virtual environment and install editable package
python -m venv venv
source venv/bin/activate
pip install -e .
```

Now any changes you make to `pyquicklook.py` are reflected immediately when you run `pyquicklook`.

## License

MIT License – see the LICENSE file for details.
