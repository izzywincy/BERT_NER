import os
import shutil
from collections import defaultdict

SOURCE_FOLDER = 'train_data'
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

    # ✅ Step 1: Force distribute CNS-tagged files first
    cns_files = [(f, c) for f, c in files_counts if c['constitutes'] > 0]
    files_counts = [(f, c) for f, c in files_counts if c['constitutes'] == 0]  # remove CNS files from general pool

    for i, (filename, counts) in enumerate(cns_files):
        split = ['train', 'eval', 'test'][i % 3]  # round-robin into each split
        split_files[split].append(filename)
        for k in ENTITY_KEYS:
            split_counts[split][k] += counts[k]

    # ✅ Step 2: Distribute remaining files with balance logic
    for filename, counts in files_counts:
        best_split = None
        min_entity_sum = float('inf')

        for split in ['train', 'eval', 'test']:
            if len(split_files[split]) < target_counts[split]:
                projected_sum = sum(
                    split_counts[split][k] + counts[k] for k in ENTITY_KEYS
                )
                if projected_sum < min_entity_sum:
                    min_entity_sum = projected_sum
                    best_split = split

        split_files[best_split].append(filename)
        for k in ENTITY_KEYS:
            split_counts[best_split][k] += counts[k]

    return split_files, split_counts


def main():
    files_counts = []
    for file in os.listdir(SOURCE_FOLDER):
        if file.endswith('.iob'):
            path = os.path.join(SOURCE_FOLDER, file)
            counts = count_in_file(path)
            files_counts.append((file, counts))

    split_files_dict, split_counts = split_files(files_counts, SPLIT_RATIOS)

    for split in ['train', 'eval', 'test']:
        split_path = os.path.join(SOURCE_FOLDER, split)
        os.makedirs(split_path, exist_ok=True)
        for file in split_files_dict[split]:
            shutil.copy(os.path.join(SOURCE_FOLDER, file), os.path.join(split_path, file))

    # Print result summary
    for split in ['train', 'eval', 'test']:
        print(f"\n📁 {split.upper()} ({len(split_files_dict[split])} files):")
        for key in ENTITY_KEYS:
            print(f"  {key}: {split_counts[split][key]}")
    print("\n✅ Stratified split complete with CNS presence in all sets!")


if __name__ == "__main__":
    main()
