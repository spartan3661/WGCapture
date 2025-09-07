# wgc_screenshot/__main__.py
from core import capture_screen
import sys
from PIL import Image
import pygetwindow as gw

def main(argv=None):
    print(gw.getAllTitles())
    img = capture_screen("TEVI")

    Image.fromarray(img, mode="RGBA").show()
    
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    raise SystemExit(main())
