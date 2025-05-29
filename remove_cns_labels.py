import os

def clean_cns_tags_in_folder(folder_path):
    updated_files = 0
    for filename in os.listdir(folder_path):
        if not filename.endswith(".iob"):
            continue

        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = []
        for line in lines:
            if line.strip() == "":
                cleaned_lines.append(line)  # preserve sentence boundary
                continue

            token, tag = line.strip().split('\t')
            if tag not in ("B-CNS", "I-CNS"):
                cleaned_lines.append(line)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        updated_files += 1

    print(f"✅ Cleaned {updated_files} files in '{folder_path}' (removed CNS labels)")

# 📂 Run on your datasets
clean_cns_tags_in_folder("queue")
