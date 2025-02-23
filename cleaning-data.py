import json
import os

# Define input and output folders
input_folder = "raw_data"  # Folder containing raw JSONL files
output_folder = "cleaned_data"  # Folder to save cleaned IOB files
error_log_folder = "error_logs"  # Folder to save error logs

# Ensure output directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(error_log_folder, exist_ok=True)

# Clear the error logs folder before starting
for error_file in os.listdir(error_log_folder):
    error_file_path = os.path.join(error_log_folder, error_file)
    if os.path.isfile(error_file_path):
        os.remove(error_file_path)  # Delete the file
print("🗑️ Cleared error logs before starting data cleaning.")

# Function to tokenize text and calculate token offsets
def tokenize_with_offsets(text):
    tokens = text.split()  # Simple whitespace tokenizer
    token_offsets = []
    start = 0
    for token in tokens:
        token_offsets.append((start, start + len(token)))
        start += len(token) + 1  # Account for space
    return tokens, token_offsets

# Function to convert JSONL annotations to IOB format
def jsonl_to_iob(data):
    text = data["text"]
    tokens, token_offsets = tokenize_with_offsets(text)
    iob_tags = ['O'] * len(tokens)

    # Extract labels from different possible formats
    labels = []
    if "label" in data:
        labels = data["label"]
    elif "labels" in data:
        labels = data["labels"]
    elif "entities" in data:
        # Convert `entities` format to standard [(start, end, label)]
        for entity in data["entities"]:
            if not all(k in entity for k in ["start_offset", "end_offset", "label"]):
                continue  # Skip malformed entities
            labels.append([entity["start_offset"], entity["end_offset"], entity["label"]])

    # Assign IOB tags based on extracted labels
    for start_offset, end_offset, label in labels:
        for i, (token_start, token_end) in enumerate(token_offsets):
            if token_start >= start_offset and token_end <= end_offset:
                if token_start == start_offset:
                    iob_tags[i] = f'B-{label}'
                else:
                    iob_tags[i] = f'I-{label}'

    # Combine tokens and IOB tags
    iob_lines = [f"{token}\t{tag}" for token, tag in zip(tokens, iob_tags)]
    return iob_lines

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jsonl"):  # Process only JSONL files
        input_file = os.path.join(input_folder, filename)
        base_filename = os.path.splitext(filename)[0]  # Remove extension for naming cleaned files
        error_log_file = os.path.join(error_log_folder, f"{filename}_errors.log")
        valid_entries = []  # Store cleaned data entries separately
        errors = []  # Store errors
        
        with open(input_file, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                try:
                    data = json.loads(line.strip())  # Parse JSON
                    
                    # Validate the structure
                    if "text" not in data:
                        errors.append(f"Line {i+1}: Missing 'text'.")
                        continue

                    # Extract labels from different possible formats
                    labels = []
                    if "label" in data:
                        labels = data["label"]
                    elif "labels" in data:
                        labels = data["labels"]
                    elif "entities" in data:
                        labels = []
                        for entity in data["entities"]:
                            if not isinstance(entity, dict) or not all(k in entity for k in ["start_offset", "end_offset", "label"]):
                                errors.append(f"Line {i+1}: Invalid entity format {entity}. Expected dictionary with 'start_offset', 'end_offset', and 'label'.")
                                continue
                            labels.append([entity["start_offset"], entity["end_offset"], entity["label"]])

                    # Validate labels
                    for label in labels:
                        if not isinstance(label, (list, tuple)) or len(label) != 3:
                            errors.append(f"Line {i+1}: Invalid label format {label}. Expected [start, end, label].")
                            continue
                    
                    # Convert to IOB format
                    iob_lines = jsonl_to_iob({"text": data["text"], "label": labels})
                    valid_entries.append(iob_lines)
                
                except json.JSONDecodeError:
                    errors.append(f"Line {i+1}: Invalid JSON format.")
        
        # Save each cleaned entry in a separate IOB file
        for index, iob_lines in enumerate(valid_entries, start=1):
            output_file = os.path.join(output_folder, f"{base_filename}_{index}.iob")
            with open(output_file, "w", encoding="utf-8") as file:
                file.write("\n".join(iob_lines) + "\n\n")  # Add a blank line between documents
        
        # Save error logs (if any)
        if errors:
            with open(error_log_file, "w", encoding="utf-8") as file:
                for error in errors:
                    file.write(error + "\n")
        
        # Summary for this file
        if valid_entries:
            print(f"✅ {filename} cleaned! {len(valid_entries)} separate IOB files created in '{output_folder}/'.")
        else:
            print(f"⚠️ No valid entries in {filename}. Check '{error_log_file}' for details.")
