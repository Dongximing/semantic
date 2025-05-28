import pickle
with open("/home/ximing/specreason/hf/math-result_left/data-500-temp0_166/generations_166.pkl", "rb") as f:
    generations = pickle.load(f)
    print(generations)