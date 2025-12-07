import json
import os
import ast
import random
import traceback 
from copy import deepcopy

ALWAYS_CRITICAL_KEYS = [
    'people_number', 
    'days', 
    'org', 
    'dest', 
    'date', 
    'visiting_city_number'
]
NOISE_RANGE = 8
BASE_IMPORTANCE = {
    'budget': 90,
    'transportation': 85, 
    'house rule': 60, 
    'room type': 55, 
    # preference constraints
    'cuisine_pref': 35,
    'rating_pref': 30,
}
GLOBAL_KEYS = ['people_number', 'days', 'budget', 'org', 'dest', 'date', 'visiting_city_number']
LOCAL_KEYS = ['transportation', 'house rule', 'room type', 'cuisine', 'rating']

# ranking calculation logic
def calculate_ranks(constraint_keys):
    if not constraint_keys:
        return {}
    
    scored_items = []
    for key in constraint_keys:
        base_score = BASE_IMPORTANCE.get(key, 50)
        noise = random.uniform(-NOISE_RANGE, NOISE_RANGE)
        final_score = base_score + noise
        scored_items.append((key, final_score))
    
    scored_items.sort(key=lambda x: x[1], reverse=True)
    
    rank_map = {}
    for rank, (k, _) in enumerate(scored_items, 1):
        rank_map[k] = rank
    return rank_map

def generate_ranks_for_item(dataset_item):
    #딕셔너리가 아니면 빈 dict return
    if not isinstance(dataset_item, dict):
        return {}

    ranks_output = {}
    turn1_keys = []
    
    # --- [Turn 1] Global Constraints ---
    for key in GLOBAL_KEYS:
        val = dataset_item.get(key)
        if val not in [None, 0, 0.0, [], {}, 'None', ""]:
            if key not in ALWAYS_CRITICAL_KEYS:
                turn1_keys.append(key)

    # --- [Turn 1] Local Constraints ---
    loc_cons_raw = dataset_item.get('local_constraint')
    
    if isinstance(loc_cons_raw, str):
        try:
            loc_cons_raw = ast.literal_eval(loc_cons_raw)
        except:
            loc_cons_raw = {}
            
    #loc_cons_raw가 반드시 dict일 때만 .get call
    if isinstance(loc_cons_raw, dict):
        for key in LOCAL_KEYS:
            local_val = loc_cons_raw.get(key)
            if local_val not in [None, "None", ""] and key not in turn1_keys:
                turn1_keys.append(key)

    ranks_output['turn_1'] = calculate_ranks(turn1_keys)

    # --- [Preference-specific Turns] ---
    # preference_constraint에 들어있는 키별로 독립적인 turn 생성
    pref_raw = dataset_item.get('preference_constraint')
    if isinstance(pref_raw, dict) and pref_raw:
        base_keys = list(turn1_keys)
        if pref_raw.get('rating') not in [None, "None", ""]:
            # rating 전용 턴: cuisine_pref는 제외
            keys = [k for k in base_keys if k != 'cuisine_pref']
            if 'rating_pref' not in keys:
                keys.append('rating_pref')
            ranks_output['turn_2_rating'] = calculate_ranks(keys)
        if pref_raw.get('cuisine') not in [None, "None", ""]:
            # cuisine 전용 턴: rating_pref는 제외
            keys = [k for k in base_keys if k != 'rating_pref']
            if 'cuisine_pref' not in keys:
                keys.append('cuisine_pref')
            ranks_output['turn_2_cuisine'] = calculate_ranks(keys)

    # --- [Turn 2+] New Constraints (한 턴씩 누적) ---
    new_constraints_list = dataset_item.get('new_constraints', [])
    if isinstance(new_constraints_list, list) and len(new_constraints_list) > 0:
        cumulative_keys = list(turn1_keys)
        for idx, nc in enumerate(new_constraints_list, start=2):  # turn_2, turn_3, ...
            added = False
            if isinstance(nc, dict):
                for k, v in nc.items():
                    if v not in [None, "None", ""] and k not in cumulative_keys:
                        if k not in ALWAYS_CRITICAL_KEYS:
                            cumulative_keys.append(k)
                            added = True
            # 빈 dict이거나 추가 키가 없으면 해당 
            # turn은 만들지 않음 (preference용 빈 new_constraints 방지)
            if added:
                ranks_output[f'turn_{idx}'] = calculate_ranks(cumulative_keys)

    return ranks_output

# 3. 파일 처리 실행 함수
def process_all_files(src_root, dest_root):
    print(f"Start processing from: {src_root}")
    print(f"Output directory: {dest_root}\n")

    for root, _, files in os.walk(src_root):
        for file in files:
            if file.endswith(".json") and not file.endswith(".jsonl"):
                if "ref_info" in file: 
                    continue

                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_root)
                dest_path = os.path.join(dest_root, rel_path)
                
                print(f"Processing: {rel_path} ...", end=" ")
                
                try:
                    with open(src_path, 'r', encoding='utf-8') as f:
                        dataset = json.load(f)
                    
                    final_data = None

                    # 구조 확인 및 처리
                    # A. 리스트 구조 [item1, item2, ...]
                    if isinstance(dataset, list):
                        processed_dataset = []
                        for i, item in enumerate(dataset):
                            # 아이템이 딕셔너리가 아닌 경우 스킵 (문자열 등)
                            if not isinstance(item, dict):
                                processed_dataset.append(item)
                                continue
                                
                            new_item = deepcopy(item)
                            new_item['constraint_ranks'] = generate_ranks_for_item(new_item)
                            processed_dataset.append(new_item)
                        final_data = processed_dataset

                    # B. 딕셔너리 구조 {"key": [item1, item2...], ...}
                    elif isinstance(dataset, dict):
                        processed_dataset = {}
                        for key, value in dataset.items():
                            if isinstance(value, list):
                                new_list = []
                                for i, item in enumerate(value):
                                    if not isinstance(item, dict):
                                        new_list.append(item)
                                        continue

                                    new_item = deepcopy(item)
                                    new_item['constraint_ranks'] = generate_ranks_for_item(new_item)
                                    new_list.append(new_item)
                                processed_dataset[key] = new_list
                            else:
                                processed_dataset[key] = value
                        final_data = processed_dataset
                    
                    else:
                        print("Skipped (Unknown JSON structure)")
                        continue

                    # Save
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    with open(dest_path, 'w', encoding='utf-8') as f:
                        json.dump(final_data, f, indent=4, ensure_ascii=False)
                    
                    print("Done.")
                    
                except Exception as e:
                    print(f"\n[FAILED] {rel_path}")
                    # 상세 에러 내용
                    traceback.print_exc()

if __name__ == "__main__":
    # 경로 설정
    # dataset/
    # INPUT_ROOT = r"C:\Users\USER\Advanced_Flex_Travel\implement\agents\evaluation\database"
    # OUTPUT_ROOT = r"C:\Users\USER\Advanced_Flex_Travel\implement\agents\evaluation\database_with_ranks"
    # process_all_files(INPUT_ROOT, OUTPUT_ROOT)
    # dataset/preference
    INPUT_ROOT = r"C:\Users\USER\Advanced_Flex_Travel\implement\agents\evaluation\database\preference"
    OUTPUT_ROOT = r"C:\Users\USER\Advanced_Flex_Travel\implement\agents\evaluation\database_with_ranks\preference"
    process_all_files(INPUT_ROOT, OUTPUT_ROOT)
