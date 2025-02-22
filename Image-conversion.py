import os
from PIL import Image

def convert_to_jpeg(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in ['.jpg', '.jpeg']:
                continue  # Only process JPEG files
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    rgb_img = img.convert("RGB")
                    # Overwrite the original file with a newly encoded JPEG
                    rgb_img.save(file_path, format="JPEG", quality=95)
                    print(f"Converted {file_path}")
            except Exception as e:
                print(f"Failed to convert {file_path}: {e}")

base_dir = r"C:\Users\moham\Desktop\Food AI\dataset\Fast Food Classification V2"
convert_to_jpeg(os.path.join(base_dir, "Train"))
convert_to_jpeg(os.path.join(base_dir, "Valid"))
convert_to_jpeg(os.path.join(base_dir, "Test"))

print("Image conversion complete.")
