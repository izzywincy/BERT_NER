import os
import random
from collections import defaultdict
from pathlib import Path

# === CONFIG ===
TRAIN_FOLDER = "train_data/train"
DEFAULT_AUG_PER_FILE = 2  # Regular files
CNS_AUG_PER_FILE = 5      # CNS-rich files

# === STEP 1: COLLECT ENTITY BANK FROM TRAINING FILES ===
def extract_entities_from_files(folder):
    entity_bank = defaultdict(set)
    for filename in os.listdir(folder):
        if not filename.endswith(".iob"):
            continue
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                token, label = parts
                if label.startswith("B-") or label.startswith("I-"):
                    entity_type = label.split("-")[1]
                    entity_bank[entity_type].add(token)
    return {k: list(v) for k, v in entity_bank.items()}

# === STEP 1.5: Count CNS in a file ===
def file_has_cns(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if "B-CNS" in line or "I-CNS" in line:
                return True
    return False

# === STEP 2: AUGMENT A SINGLE FILE ===
def augment_file(file_path, entity_bank, aug_id):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    augmented_lines = []
    for line in lines:
        if line.strip() == "":
            augmented_lines.append("\n")
            continue
        parts = line.strip().split()
        if len(parts) != 2:
            augmented_lines.append(line)
            continue
        token, label = parts
        if label.startswith("B-") or label.startswith("I-"):
            entity_type = label.split("-")[1]
            if entity_type in entity_bank and entity_bank[entity_type]:
                token = random.choice(entity_bank[entity_type])
        augmented_lines.append(f"{token}\t{label}\n")

    base_filename = Path(file_path).stem
    new_filename = f"{base_filename}_aug{aug_id}.iob"
    new_filepath = os.path.join(Path(file_path).parent, new_filename)
    with open(new_filepath, "w", encoding="utf-8") as f:
        f.writelines(augmented_lines)

# === STEP 3: MAIN DRIVER ===
def main():
    entity_bank = extract_entities_from_files(TRAIN_FOLDER)
    print(f"Collected entities from {len(entity_bank)} types.")

    for filename in os.listdir(TRAIN_FOLDER):
        if not filename.endswith(".iob"):
            continue
        file_path = os.path.join(TRAIN_FOLDER, filename)
        aug_rounds = CNS_AUG_PER_FILE if file_has_cns(file_path) else DEFAULT_AUG_PER_FILE
        for aug_id in range(1, aug_rounds + 1):
            augment_file(file_path, entity_bank, aug_id)

    print(f"✅ Data augmentation completed. Augmented files saved in '{TRAIN_FOLDER}'.")

if __name__ == "__main__":
    main()
