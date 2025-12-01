import os
import pandas as pd
import shutil

CSV_PATH = "train.csv"
IMAGES_PATH = "train_images"
OUTPUT_PATH = "dataset/train"

df = pd.read_csv(CSV_PATH)

# Create folders 0 to 4
for i in range(5):
    os.makedirs(os.path.join(OUTPUT_PATH, str(i)), exist_ok=True)

# Move images
for idx, row in df.iterrows():
    file = f"{row['id_code']}.png"
    label = str(row['diagnosis'])

    src = os.path.join(IMAGES_PATH, file)
    dst = os.path.join(OUTPUT_PATH, label, file)

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied: {file} → class {label}")
    else:
        print(f"Missing: {file}")

print("DONE SORTING!!!")
