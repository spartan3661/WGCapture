# wgcapture

> Consistent, window-based screen captures on Windows using WinRT + Windows Graphics Capture (WGC).

A small library that makes it easy to capture screenshots of *Windows (application windows)* rather than whole monitors. This helps produce consistent, high-fidelity captures across multiple displays, scaling/DPI settings, and systems where monitor-based grabs are unreliable.

> **Note:** This package targets **Microsoft Windows** and uses Windows-specific APIs (WinRT + Windows Graphics Capture). It will not work on macOS or Linux.

---

## Features

* Capture a specific application window by title
* Produce a numpy.ndarray
    dtype: np.uint8(0-255)
    shape: (height, width, 4)
    channels: RGBA
* Lightweight, dependency-minimal wrapper around WinRT/WGC functionality
* Works reliably across high-DPI displays and mixed-scaling setups

---

## Installation

Install from PyPI:

```powershell
pip install wgcapture
```

Or install from TestPyPI:

```powershell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple wgcapture==0.1.0
```

**Requirements**

* Windows 10 / 11 (WinRT and Windows Graphics Capture support)
* Python 3.8+

```powershell
pip install Pillow
```

---

## Quickstart

> The code below is a ready-to-edit example.

```python
from wgcapture import capture_screen
from PIL import Image
# Capture by window title (first match)
img = capture_screen(screen="Untitled - Notepad")
Image.fromarray(img, mode="RGBA").show()

```

## API


* `capture_screen(screen: str = None)`

---

## Contributing

Contributions welcome! Please follow these guidelines:

1. Open an issue to discuss significant changes.
2. Branch from `main` using `feature/<short-desc>`.
3. Keep commits small and focused. Add tests for bug fixes/features.
4. Open a PR and request review.

Add a `CONTRIBUTING.md` with more specifics if you want.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## Acknowledgements

Built on top of WinRT and Windows Graphics Capture APIs.