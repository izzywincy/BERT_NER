# THIS FILE CLEANS THE RAW DATA (raw_data folder
import json
import os

# Define input and output folders
input_folder = "raw_data"  # Folder containing raw JSONL files
output_folder = "cleaned_data"  # Folder to save cleaned files
error_log_folder = "error_logs"  # Folder to save error logs

# Ensure output directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(error_log_folder, exist_ok=True)

# Clear the error logs folder before starting
for error_file in os.listdir(error_log_folder):
    error_file_path = os.path.join(error_log_folder, error_file)
    if os.path.isfile(error_file_path):
        os.remove(error_file_path)  # Delete the file
print("🗑️ Error logs were cleared.")

# Get list of already cleaned files 
cleaned_files = set(os.listdir(output_folder))

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jsonl"):  # Process only JSONL files
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        error_log_file = os.path.join(error_log_folder, f"{filename}_errors.log")


        # Check if file is already cleaned
        if filename in cleaned_files:
            print(f"🔁 {filename} is already cleaned. Skipping...")
            continue

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

                    entities = []

                    # Handle first structure: "label": [[start, end, label]]
                    if "label" in data:
                        for label in data["label"]:
                            if not isinstance(label, (list, tuple)) or len(label) != 3:
                                errors.append(f"Line {i+1}: Invalid label format {label}. Expected [start, end, label].")
                                continue
                            start, end, entity_label = label
                            entities.append({"start": start, "end": end, "label": str(entity_label)})

                    # Handle second structure: "labels": [[start, end, label]]
                    elif "labels" in data:
                        for label in data["labels"]:
                            if not isinstance(label, (list, tuple)) or len(label) != 3:
                                errors.append(f"Line {i+1}: Invalid label format {label}. Expected [start, end, label].")
                                continue
                            start, end, entity_label = label
                            entities.append({"start": start, "end": end, "label": str(entity_label)})


                    # Handle second structure: "entities": [{"id": X, "label": "ENTITY", "start_offset": A, "end_offset": B}]
                    elif "entities" in data:
                        for entity in data["entities"]:
                            if not isinstance(entity, dict) or not all(k in entity for k in ["start_offset", "end_offset", "label"]):
                                errors.append(f"Line {i+1}: Invalid entity format {entity}. Expected dictionary with 'start_offset', 'end_offset', and 'label'.")
                                continue
                            entities.append({
                                "start": entity["start_offset"],
                                "end": entity["end_offset"],
                                "label": str(entity["label"])
                            })

                    else:
                        errors.append(f"Line {i+1}: No 'labels' or 'entities' field found.")
                        continue

                    # Store new JSON structure
                    fixed_entry = {
                        "text": data["text"],
                        "entities": entities  # Standardized format
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
