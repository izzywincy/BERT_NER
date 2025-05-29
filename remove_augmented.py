# Delete augmented files from a directory

import os

def remove_augmented_files(folder_path):
    removed_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".iob") and "_aug" in filename:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            removed_count += 1
            print(f"🗑️ Removed: {filename}")
    
    print(f"\n✅ Done. Removed {removed_count} augmented files from '{folder_path}'.")

# 🔁 Use this on your dataset folders:
remove_augmented_files("queue")
