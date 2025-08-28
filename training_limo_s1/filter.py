import os
import shutil
import json
from tqdm import tqdm

src_dir = './data_s1_math_qwq'
dst_dir = './data_s1_200_math_qwq'
os.makedirs(dst_dir, exist_ok=True)

# 遍历所有子文件夹
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
        # 筛选 segment 数量小于 100
        if len(data) < 200:
            # 拷贝整个子文件夹
            dst_subdir = os.path.join(dst_dir, subdir)
            if os.path.exists(dst_subdir):
                shutil.rmtree(dst_subdir)
            shutil.copytree(src_subdir, dst_subdir)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")

print("筛选与拷贝完成！")
