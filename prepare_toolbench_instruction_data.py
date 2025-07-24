import os
import json
from sklearn.model_selection import train_test_split

INSTRUCTION_DIR = "/home/jhlee/ToolBench/data/instruction"
SAVE_DIR = "data"
SPLITS = ["G1_query.json", "G2_query.json", "G3_query.json"]

os.makedirs(SAVE_DIR, exist_ok=True)

all_samples = []
for split in SPLITS:
    file_path = os.path.join(INSTRUCTION_DIR, split)
    print(f"Loading {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            query = sample.get("query", "")
            api_list = sample.get("api_list", [])
            relevant_api_keys = set()
            for rel in sample.get("relevant APIs", []):
                if len(rel) == 2:
                    relevant_api_keys.add((rel[0], rel[1]))  # (tool_name, api_name)
            relevant_apis = [api for api in api_list if (api.get("tool_name"), api.get("api_name")) in relevant_api_keys]
            irrelevant_apis = [api for api in api_list if (api.get("tool_name"), api.get("api_name")) not in relevant_api_keys]
            all_samples.append({
                "query": query,
                "relevant_apis": relevant_apis,
                "irrelevant_apis": irrelevant_apis,
                "api_list": api_list,
                "query_id": sample.get("query_id", None)
            })
print(f"Loaded {len(all_samples)} samples from ToolBench instruction data.")

# Split into train/eval/test (8:1:1)
train, temp = train_test_split(all_samples, test_size=0.2, random_state=42)
eval, test = train_test_split(temp, test_size=0.5, random_state=42)

with open(os.path.join(SAVE_DIR, "train.json"), "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open(os.path.join(SAVE_DIR, "eval.json"), "w", encoding="utf-8") as f:
    json.dump(eval, f, ensure_ascii=False, indent=2)
with open(os.path.join(SAVE_DIR, "test.json"), "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)
print(f"Saved splits to {SAVE_DIR}/train.json, eval.json, test.json") 