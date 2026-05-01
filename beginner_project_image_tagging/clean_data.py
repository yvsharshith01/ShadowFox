from PIL import Image
import os

dataset_path = "data"

bad_files = []
count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        count += 1

        # Show progress every 500 images
        if count % 500 == 0:
            print(f"Checked {count} images...")

        try:
            with Image.open(file_path) as img:
                img.load()   # Faster and safer
        except Exception:
            print("Removing:", file_path)
            bad_files.append(file_path)

# Delete corrupted files
for file in bad_files:
    os.remove(file)

print(f"\n Removed {len(bad_files)} corrupted files.")