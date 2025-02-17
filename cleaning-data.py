# THIS FILE CLEANS THE RAW DATA (raw_data folder)
import json
import os

# Define input and output folders
input_folder = "raw_data"  # Folder containing raw JSONL files
output_folder = "cleaned_data"  # Folder to save cleaned files
error_log_folder = "error_logs"  # Folder to save error logs

# Ensure output directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(error_log_folder, exist_ok=True)

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jsonl"):  # Process only JSONL files
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        error_log_file = os.path.join(error_log_folder, f"{filename}_errors.log")

        valid_data = []  # Store cleaned data
        errors = []  # Store errors

        with open(input_file, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                try:
                    data = json.loads(line.strip())  # Parse JSON

                    # Validate the structure
                    if "text" not in data:
                        errors.append(f"Line {i+1}: Missing 'text'.")
                        continue

                    # Handle both "label" (single) and "labels" (list)
                    raw_labels = data.get("labels") or data.get("label")
                    
                    if raw_labels is None:
                        errors.append(f"Line {i+1}: Missing 'labels' or 'label'.")
                        continue

                    # Ensure raw_labels is a list (convert if necessary)
                    if isinstance(raw_labels, dict):
                        errors.append(f"Line {i+1}: Invalid 'labels' format (dictionary found). Expected list.")
                        continue
                    elif not isinstance(raw_labels, list):
                        raw_labels = [raw_labels]  # Convert single label to a list

                    # Convert labels to 'entities' format
                    entities = []
                    for label in raw_labels:
                        if not isinstance(label, (list, tuple)) or len(label) != 3:
                            errors.append(f"Line {i+1}: Invalid label format {label}. Expected [start, end, label].")
                            continue
                        start, end, entity_label = label
                        entities.append({"start": start, "end": end, "label": str(entity_label)})  # Ensure label is a string

                    # Store new JSON structure
                    fixed_entry = {
                        "text": data["text"],
                        "entities": entities  # Replacing 'labels' or 'label' with formatted 'entities'
                    }
                    valid_data.append(fixed_entry)

                except json.JSONDecodeError:
                    errors.append(f"Line {i+1}: Invalid JSON format.")

        # Save cleaned data
        with open(output_file, "w", encoding="utf-8") as file:
            for entry in valid_data:
                file.write(json.dumps(entry) + "\n")

        # Save error logs (if any)
        if errors:
            with open(error_log_file, "w", encoding="utf-8") as file:
                for error in errors:
                    file.write(error + "\n")

        # Summary for this file
        print(f"✅ {filename} cleaned! {len(valid_data)} valid entries saved in '{output_file}'.")
        print(f"⚠️ {len(errors)} errors found. Check '{error_log_file}' for details.")
