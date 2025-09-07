import ctypes as C
from ctypes import c_int32, c_uint8, c_void_p, POINTER
from ctypes.wintypes import HWND as C_HWND
import numpy as np
import pygetwindow as gw
from PIL import Image
from importlib import resources
import os, sys, struct, platform, threading
from typing import cast


_dll = None
_dll_initialized = False
_dll_lock = threading.Lock()
_dll_dir_handle = None

PACKAGE_NAME = __package__ or "wgc_screenshot"

class BGRAFrame(C.Structure):
    _fields_ = [
        ("data",   C.POINTER(c_uint8)),
        ("width",  c_int32),
        ("height", c_int32),
        ("stride", c_int32),
        ("size",   c_int32),
    ]


def _is_windows() -> bool:
    return sys.platform == "win32"

def _is_process_64bit() -> bool:
    # pointer size in bits
    return struct.calcsize("P") * 8 == 64

def is_supported_platform() -> bool:
    return _is_windows() and _is_process_64bit()

def _resolve_dll_path():
    try:
        dll_path = resources.files(PACKAGE_NAME).joinpath("dlls").joinpath("wgc_core.dll")
        with resources.as_file(dll_path) as p:
            return str(p)
    except Exception:
        this_dir = os.path.dirname(__file__)
        return os.path.join(this_dir, "dlls", "wgc_core.dll")

def _ensure_dll():
    global _dll, _dll_initialized, _dll_dir_handle

    if _dll_initialized and _dll is not None:
        return

    with _dll_lock:
        if _dll_initialized and _dll is not None:
            return

        if not _is_windows() or not _is_process_64bit():
            raise RuntimeError(
                f"wgc_screenshot only supports Windows x64. "
                f"Detected: {platform.system()} {struct.calcsize('P')*8}-bit Python"
            )

        dll_path = _resolve_dll_path()
        if not os.path.exists(dll_path):
            raise RuntimeError(f"wgc_core.dll not found at {dll_path!r}")

        dll_dir = os.path.dirname(dll_path)
        try:
            if hasattr(os, "add_dll_directory"):

                _dll_dir_handle = os.add_dll_directory(dll_dir)
        except Exception:
            _dll_dir_handle = None

        try:
            _dll = C.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load wgc_core.dll from {dll_path!r}: {e}") from e

        # configure argtypes/restype once
        _dll.wgc_capture_bgra.argtypes = [C_HWND, C.POINTER(BGRAFrame), c_int32]
        _dll.wgc_capture_bgra.restype  = c_int32
        _dll.wgc_free.argtypes = [c_void_p]
        _dll.wgc_free.restype  = None

        _dll_initialized = True

def _get_dll():
    _ensure_dll()
    return _dll

def _capture_hwnd_to_image(hwnd: int, timeout_ms=2000):
    frame = BGRAFrame()
    dll = _get_dll()
    dll = cast(C.CDLL, _get_dll())
    rc = dll.wgc_capture_bgra(C_HWND(hwnd), C.byref(frame), timeout_ms)
    if rc != 0:
        raise RuntimeError(f"capture failed rc={rc}")

    try:
        byte_count = frame.stride * frame.height
        buf = C.cast(frame.data, C.POINTER(c_uint8 * byte_count)).contents
        np_rowbuf = np.frombuffer(buf, dtype=np.uint8).reshape(frame.height, frame.stride)
        np_bgra = np_rowbuf[:, :frame.width * 4].copy()
        return np_bgra
    finally:
        dll.wgc_free(frame.data)

# DEBUG
def _screen_list():
    return gw.getAllTitles()

def capture_screen(screen):
    """
    Returns a numpy.ndarray
    dtype: np.uint8(0-255)
    shape: (height, width, 4)
    channels: RGBA
    """
    target_window = gw.getWindowsWithTitle(screen)[0]

    if not target_window:
        raise RuntimeError(f"No window matching: {target_window}")
    
    hwnd = target_window._hWnd

    # capture_hwnd_to_image returns 2D (H, width*4). reshape to (H, W, 4)
    bgra = _capture_hwnd_to_image(hwnd)
    arr = np.asarray(bgra, dtype=np.uint8)

    if arr.ndim == 2:
        h, row_bytes = arr.shape
        if row_bytes % 4 != 0:
            raise ValueError("row byte count not divisible by 4")
        w = row_bytes // 4
        img_bgra = arr.reshape((h, w, 4))
    elif arr.ndim == 3:
        img_bgra = arr
    else:
        raise ValueError(f"unexpected bgra ndim={arr.ndim}")
    
    img_bgra = np.ascontiguousarray(img_bgra)

    rgba = np.ascontiguousarray(img_bgra[..., [2, 1, 0, 3]])
    return rgba


