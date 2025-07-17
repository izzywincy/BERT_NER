import os
import shutil
from collections import defaultdict

QUEUE_FOLDER = 'queue'  # Input: all raw + augmented files
OUTPUT_ROOT = 'train_data'         # Output: where train/eval/test folders are
SPLIT_RATIOS = {'train': 0.7, 'eval': 0.2, 'test': 0.1}
ENTITY_KEYS = ['case_nums', 'persons', 'institutions', 'prom_dates', 'republic_acts', 'statutes', 'constitutes']


def count_in_file(file_path):
    counters = dict.fromkeys(ENTITY_KEYS, 0)
    entity_map = {
        'CASE_NUM': 'case_nums', 'PERSON': 'persons', 'INS': 'institutions',
        'PROM_DATE': 'prom_dates', 'RA': 'republic_acts', 'STA': 'statutes', 'CNS': 'constitutes'
    }

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            token, tag = parts
            for ent_type, key in entity_map.items():
                if tag == f'B-{ent_type}':
                    counters[key] += 1
    return counters


def sum_entities(entity_dicts):
    total = dict.fromkeys(ENTITY_KEYS, 0)
    for d in entity_dicts:
        for key in ENTITY_KEYS:
            total[key] += d[key]
    return total


def split_files(files_counts, ratios):
    files_counts.sort(key=lambda x: sum(x[1].values()), reverse=True)

    split_files = {'train': [], 'eval': [], 'test': []}
    split_counts = {'train': defaultdict(int), 'eval': defaultdict(int), 'test': defaultdict(int)}

    total_files = len(files_counts)
    target_counts = {
        'train': int(ratios['train'] * total_files),
        'eval': int(ratios['eval'] * total_files),
        'test': total_files - int(ratios['train'] * total_files) - int(ratios['eval'] * total_files)
    }

    # ✅ Separate CNS-tagged files
    cns_files = [(f, c) for f, c in files_counts if c['constitutes'] > 0]
    non_cns_files = [(f, c) for f, c in files_counts if c['constitutes'] == 0]

    # ✅ Put 90% of CNS files in train, 10% in eval
    cns_train_count = int(0.9 * len(cns_files))
    for i, (filename, counts) in enumerate(cns_files):
        split = 'train' if i < cns_train_count else 'eval'
        split_files[split].append(filename)
        for k in ENTITY_KEYS:
            split_counts[split][k] += counts[k]

    # 🔄 Distribute remaining files with balance logic
    for filename, counts in non_cns_files:
        best_split = None
        min_entity_sum = float('inf')

        for split in ['train', 'eval', 'test']:
            if len(split_files[split]) < target_counts[split]:
                projected_sum = sum(split_counts[split][k] + counts[k] for k in ENTITY_KEYS)
                if projected_sum < min_entity_sum:
                    min_entity_sum = projected_sum
                    best_split = split

        split_files[best_split].append(filename)
        for k in ENTITY_KEYS:
            split_counts[best_split][k] += counts[k]

    return split_files, split_counts


def clear_previous_splits():
    for split in ['train', 'eval', 'test']:
        split_path = os.path.join(OUTPUT_ROOT, split)
        if os.path.exists(split_path):
            for f in os.listdir(split_path):
                os.remove(os.path.join(split_path, f))
        else:
            os.makedirs(split_path)


def main():
    clear_previous_splits()

    files_counts = []
    for file in os.listdir(QUEUE_FOLDER):
        file_path = os.path.join(QUEUE_FOLDER, file)
        if file.endswith('.iob') and os.path.isfile(file_path):
            counts = count_in_file(file_path)
            files_counts.append((file, counts))

    split_files_dict, split_counts = split_files(files_counts, SPLIT_RATIOS)

    for split in ['train', 'eval', 'test']:
        split_path = os.path.join(OUTPUT_ROOT, split)
        for file in split_files_dict[split]:
            shutil.copy(os.path.join(QUEUE_FOLDER, file), os.path.join(split_path, file))

    # Summary
    for split in ['train', 'eval', 'test']:
        print(f"\n📁 {split.upper()} ({len(split_files_dict[split])} files):")
        for key in ENTITY_KEYS:
            print(f"  {key}: {split_counts[split][key]}")
    print("\n✅ Stratified re-split complete!")


if __name__ == "__main__":
    main()
