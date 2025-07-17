import os
import re

def clean_augmented_files(queue_folder):
    """
    Removes files ending with 'aug.iob' and 'aug{N}.iob' where N > 2.
    Keeps 'aug1.iob', 'aug2.iob', and all non-augmented files.
    """
    pattern = re.compile(r'aug(\d*)\.iob$')  # Match 'aug' followed by optional digits

    for filename in os.listdir(queue_folder):
        print(f"Checking file: {filename}")  # Debugging line to print filenames
        match = pattern.search(filename)
        if match:
            aug_num_str = match.group(1)
            print(f"Match found! aug_num_str: {aug_num_str}")  # Debugging line
            file_path = os.path.join(queue_folder, filename)
            print(f"🗑️ Removing: {filename}")
            os.remove(file_path)
            print(f"✅ Removed: {filename}")

clean_augmented_files('queue')  # Replace 'queue' with your actual queue folder path
print("✅ Excess augmented files removed successfully!")
