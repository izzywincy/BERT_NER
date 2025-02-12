# THIS FILE CLEANS THE RAW DATA (data.jsonl) 
import json

# File paths
input_file = "data.jsonl"  # Your original file
output_file = "fixed_data.jsonl"  # The cleaned file

valid_data = []  # Store cleaned data
errors = []  # Store errors

with open(input_file, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        try:
            data = json.loads(line.strip())  # Parse JSON
            
            # Validate the structure
            if "text" not in data or "labels" not in data:
                errors.append(f"Line {i+1}: Missing 'text' or 'labels'.")
                continue

            # Convert 'labels' list to 'entities' list
            entities = []
            for label in data["labels"]:
                if len(label) != 3:
                    errors.append(f"Line {i+1}: Invalid label format {label}. Expected [start, end, label].")
                    continue
                start, end, entity_label = label
                entities.append({"start": start, "end": end, "label": str(entity_label)})  # Ensure label is a string

            # Store new JSON structure
            fixed_entry = {
                "text": data["text"],
                "entities": entities  # Replacing 'labels' with formatted 'entities'
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
    with open("errors.log", "w", encoding="utf-8") as file:
        for error in errors:
            file.write(error + "\n")

# Summary
print(f"✅ Cleaning complete! {len(valid_data)} valid entries saved in '{output_file}'.")
print(f"⚠️ {len(errors)} errors found. Check 'errors.log' for details.")
