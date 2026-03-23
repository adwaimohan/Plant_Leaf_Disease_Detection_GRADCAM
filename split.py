import os
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = r"C:/Users/KIIT0001/Desktop/work/projects/crop_disease_detection/dataset/PlantVillage"
OUTPUT_DIR  = r"C:/Users/KIIT0001/Desktop/work/projects/crop_disease_detection/dataset/PlantVillage"

TEST_SIZE = 0.10
VAL_SIZE = 0.10

classes = [c for c in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, c))]

for cls in classes:
    class_path = os.path.join(DATASET_DIR, cls)
    
    images = [img for img in os.listdir(class_path)
              if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(images) == 0:
        print(f"Skipping {cls} — no images found.")
        continue

    train_imgs, test_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=42)
    train_imgs, val_imgs  = train_test_split(train_imgs, test_size=VAL_SIZE, random_state=42)

    for split_name, split_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        split_dir = os.path.join(OUTPUT_DIR, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_list:
            shutil.copy(os.path.join(class_path, img), os.path.join(split_dir, img))

print("\nDone — dataset successfully split!")
