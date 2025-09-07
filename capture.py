# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Zongyu Carnes
import ctypes as C
from ctypes import c_int32, c_uint8, c_void_p, POINTER
from ctypes.wintypes import HWND as C_HWND
import numpy as np
import pygetwindow as gw
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL_PATH = os.path.join(BASE_DIR, "Python Screenshot Library", "wgc_core.dll")
dll = C.CDLL(DLL_PATH)

class BGRAFrame(C.Structure):
    _fields_ = [
        ("data",   C.POINTER(c_uint8)),
        ("width",  c_int32),
        ("height", c_int32),
        ("stride", c_int32),
        ("size",   c_int32),
    ]

dll.wgc_capture_bgra.argtypes = [C_HWND, C.POINTER(BGRAFrame), c_int32]
dll.wgc_capture_bgra.restype  = c_int32
dll.wgc_free.argtypes = [c_void_p]
dll.wgc_free.restype  = None

def capture_hwnd_to_image(hwnd: int, timeout_ms=2000):
    frame = BGRAFrame()
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
def screen_list():
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
    bgra = capture_hwnd_to_image(hwnd)
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

def main():
    print(gw.getAllTitles())
    img = capture_screen("TEVI")


    Image.fromarray(img, mode="RGBA").show()
    
if __name__ == "__main__":
    main()
