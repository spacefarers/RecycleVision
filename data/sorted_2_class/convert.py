import os
from PIL import Image

def convert_png_to_jpg(root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(".png"):
                png_path = os.path.join(dirpath, fname)
                jpg_path = os.path.splitext(png_path)[0] + ".jpg"

                try:
                    img = Image.open(png_path).convert("RGB")
                    img.save(jpg_path, "JPEG", quality=95)
                    print("Converted:", png_path, "->", jpg_path)
                except Exception as e:
                    print("Failed converting", png_path, "error:", e)


if __name__ == "__main__":
    convert_png_to_jpg(".")

