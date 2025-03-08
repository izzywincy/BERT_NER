from transformers import BertForTokenClassification, AutoTokenizer
import torch
import os
from collections import Counter
from sklearn.metrics import classification_report

# ------------------ Step 1: Load the Trained Model & Tokenizer ------------------
model_path = "./bert-legal-ner"
if not os.path.exists(model_path):
    raise ValueError("⚠️ Model directory not found! Ensure the model is trained and saved.")

model = BertForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()
print("✅ Model and tokenizer loaded successfully!")

# ------------------ Step 2: Load Annotated Test Set from Folder ------------------
def load_iob_test_set(folder_path):
    tokens, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".iob"):  # Ensure we only process IOB files
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                token_list, label_list = [], []
                for line in f:
                    line = line.strip()
                    if line:
                        token, label = line.split("\t")
                        token_list.append(token)
                        label_list.append(label)
                    else:
                        tokens.append(token_list)
                        labels.append(label_list)
                        token_list, label_list = [], []
                if token_list:
                    tokens.append(token_list)
                    labels.append(label_list)
    return tokens, labels

test_folder = "./test_data"  # Change to the actual test set folder
test_tokens, test_labels = load_iob_test_set(test_folder)
print(f"✅ Loaded {len(test_tokens)} test sentences from folder.")

# ------------------ Step 3: Run Model Inference ------------------
tokenized_inputs = tokenizer(
    test_tokens, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**tokenized_inputs)

logits = outputs.logits
predictions = torch.argmax(logits, dim=2)

# ------------------ Step 4: Convert Predictions to Labels ------------------
id2label = model.config.id2label

def align_predictions(predictions, tokenized_inputs):
    aligned_labels = []
    for batch_idx, pred in enumerate(predictions):
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        aligned_preds = []
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            aligned_preds.append(id2label[pred[idx].item()])
            previous_word_idx = word_idx
        aligned_labels.append(aligned_preds)
    return aligned_labels

cleaned_labels = align_predictions(predictions, tokenized_inputs)

# ------------------ Step 5: Compute Evaluation Metrics ------------------
true_labels = [label for sent_labels in test_labels for label in sent_labels]
pred_labels = [label for sent_labels in cleaned_labels for label in sent_labels]

# Ensure consistent length
min_length = min(len(true_labels), len(pred_labels))
true_labels = true_labels[:min_length]
pred_labels = pred_labels[:min_length]

print("\n🔹 Evaluation Metrics:")
print(classification_report(true_labels, pred_labels, digits=4))

# ------------------ Step 6: Print Sample Output ------------------
print("\n🔹 Sample NER Output:")
for token, label in zip(test_tokens[0], cleaned_labels[0]):
    print(f"{token}: {label}")