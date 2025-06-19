from transformers import AutoTokenizer

import os
import json
from transformers import AutoTokenizer

# 配置参数
root_dir = "/data/semantic/speculative/spec_result_math-500_seed_42"
prefix = "spec_math-500_"
json_file = "spec_generation.json"
tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-AWQ")  # 替换为你的tokenizer名

for dirname in os.listdir(root_dir):
    if dirname.startswith(prefix):
        dirpath = os.path.join(root_dir, dirname)
        json_path = os.path.join(dirpath, json_file)
        if not os.path.exists(json_path):
            print(f"文件不存在: {json_path}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 只处理 data[0]['real_answer']
        try:
            real_answer = data[0]['real_answer']
            tokens = tokenizer.encode(real_answer,return_tensors="pt")
            length = tokens.shape[1]
            data[0]['length_of_real_output'] = length
        except Exception as e:
            print(f"处理 {json_path} 时出错：{e}")
            continue

        # 写回原文件
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"处理完成: {json_path}")
