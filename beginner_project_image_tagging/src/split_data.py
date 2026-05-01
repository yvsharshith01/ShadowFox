import os
import shutil
import random

#  CHANGE THIS TO YOUR DATASET PATH
original_dataset_dir = r"C:\Users\Lenovo\Downloads\archive (1)"

# Output base directory (inside your project)
base_dir = "data"

# Split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2



cat_src = os.path.join(original_dataset_dir, "cat")
dog_src = os.path.join(original_dataset_dir, "dog")

# Destination folders
train_cat = os.path.join(base_dir, "train", "cat")
train_dog = os.path.join(base_dir, "train", "dog")

val_cat = os.path.join(base_dir, "val", "cat")
val_dog = os.path.join(base_dir, "val", "dog")

test_cat = os.path.join(base_dir, "test", "cat")
test_dog = os.path.join(base_dir, "test", "dog")

# Create all directories
for path in [train_cat, train_dog, val_cat, val_dog, test_cat, test_dog]:
    os.makedirs(path, exist_ok=True)



def split_and_copy(src_folder, train_dir, val_dir, test_dir):
    print(f"\nProcessing: {src_folder}")

    # Check if source exists
    if not os.path.exists(src_folder):
        print(f" ERROR: Source folder not found -> {src_folder}")
        return

    images = os.listdir(src_folder)
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    print(f"Total: {total} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    # Copy function
    def copy_files(file_list, dest_folder):
        for img in file_list:
            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(dest_folder, img)

            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f" Error copying {img}: {e}")
            else:
                print(f" Missing file: {src_path}")

    # Copy to respective folders
    copy_files(train_imgs, train_dir)
    copy_files(val_imgs, val_dir)
    copy_files(test_imgs, test_dir)



if __name__ == "__main__":
    print(" Starting dataset split...")

    split_and_copy(cat_src, train_cat, val_cat, test_cat)
    split_and_copy(dog_src, train_dog, val_dog, test_dog)

    print("\n Dataset split completed successfully!")