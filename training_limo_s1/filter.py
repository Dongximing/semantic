import os
import shutil
import json
from tqdm import tqdm

src_dir = './data_s1_science_qwq'
dst_dir = './data_s1_200_segments_science_qwq'
os.makedirs(dst_dir, exist_ok=True)

# Iterate through all subdirectories.
for subdir in tqdm(sorted(os.listdir(src_dir))):
    src_subdir = os.path.join(src_dir, subdir)
    if not os.path.isdir(src_subdir):
        continue
    json_path = os.path.join(src_subdir, "generation.json")
    if not os.path.exists(json_path):
        continue
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Keep subdirectories with fewer than 200 segments.
        if len(data) < 200:
            # Copy the whole subdirectory.
            dst_subdir = os.path.join(dst_dir, subdir)
            if os.path.exists(dst_subdir):
                shutil.rmtree(dst_subdir)
            shutil.copytree(src_subdir, dst_subdir)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")

print("Filtering and copying completed.")
