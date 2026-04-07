"""
Person 1, Step 6: Run enhancement pipeline on images in test_images/.
Run from project root: python scripts/test_enhancement.py
"""
import os
import sys

# Run from project root so config and enhancement resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhancement import enhance_image

TEST_DIR = "test_images"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

def main():
    if not os.path.isdir(TEST_DIR):
        print(f"Folder {TEST_DIR}/ not found. Create it and add 2–3 test images.")
        return
    names = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(EXTENSIONS)]
    if not names:
        print(f"Add 2–3 images to {TEST_DIR}/ then run this script again.")
        return
    for name in sorted(names):
        path = os.path.join(TEST_DIR, name)
        out = enhance_image(path)
        if out:
            print(f"Enhanced: {path} -> {out}")
        else:
            print(f"Failed to load: {path}")
    print("Check results in lighting_enhancement/output/")

if __name__ == "__main__":
    main()
