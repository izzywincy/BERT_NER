# THIS FILE DIRECTLY TESTS THE MODEL WITH INPUT [TEXT] DATA
# ALSO SHOWS THE DISTRIBUTION OF THE TRAINING DATA 

from transformers import BertForTokenClassification, BertTokenizer
import torch
import os
from collections import Counter

# ------------------ Step 1: Load the Trained Model & Tokenizer ------------------
# Check if trained model exists
model_path = "./bert-legal-ner"
if not os.path.exists(model_path):
    raise ValueError("⚠️ Model directory not found! Ensure the model is trained and saved.")

# Load trained model & tokenizer
model = BertForTokenClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()
print("✅ Model and tokenizer loaded successfully!")

# ------------------ Step 2: Read IOB Data and Analyze Label Distribution ------------------
dataset_path = "./cleaned_data"  # Adjust if your IOB files are in a different location
all_labels = []

# Read all .iob files from the dataset folder
for filename in os.listdir(dataset_path):
    if filename.endswith(".iob"):  # Ensure we only process IOB files
        with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Ignore empty lines
                    token, label = line.split()  # Split token and label
                    all_labels.append(label)

# Print class distribution
label_counts = Counter(all_labels)
print("\n🔍 Label Distribution in Training Data:", label_counts)

# ------------------ Step 3: Run a Test Sentence ------------------
test_texts = ["The Supreme Court ruled under Republic Act No. 3019."]
inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

# Run model inference
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to predictions
logits = outputs.logits
predictions = torch.argmax(logits, dim=2)

# Print raw logits for debugging
print("\n🔍 Logits (first 5 tokens):", logits[0][:5])  # Show first 5 token logits

# ------------------ Step 4: Convert Predictions Back to Labels ------------------
id2label = model.config.id2label  # Use the model's stored label map

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [id2label[p.item()] for p in predictions[0]]

# Print results
print("\n🔹 Sentence: ", test_texts[0])
print("🔹 Predictions:")
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")

# ------------------ Step 5: Debugging - If Model Predicts Only "O" ------------------
if all(label == "O" for label in predicted_labels):
    print("\n⚠️ WARNING: Model is predicting 'O' for all tokens. Possible causes:")
    print("- Dataset might be imbalanced (check label distribution).")
    print("- Model might not have trained properly (increase epochs, reduce warmup_steps).")
    print("- Logits might be too close to zero (model is not confident in predictions).")
