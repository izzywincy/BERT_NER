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

# ------------------ Step 2: Load Annotated Test Set ------------------
def load_iob_file(file_path):
    tokens, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
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

test_folder = "./train_data/test"  # Change to the actual test set folder
test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".iob")]

id2label = model.config.id2label

# ------------------ Step 3: Process Each Test File Separately ------------------
for file_path in test_files:
    filename = os.path.basename(file_path)
    test_tokens, test_labels = load_iob_file(file_path)
    print(f"\n📌 Processing File: {filename}")
    print(f"✅ Successfully Loaded {filename}!")

    # ------------------ Step 4: Run Model Inference ------------------
    tokenized_inputs = tokenizer(
        test_tokens, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    # ------------------ Step 5: Convert Predictions to Labels ------------------
    def align_predictions(predictions, tokenized_inputs):
        aligned_labels = []
        for batch_idx, pred in enumerate(predictions):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_idx)
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

    # ------------------ Step 6: Compute Evaluation Metrics ------------------
    true_labels = [label for sent_labels in test_labels for label in sent_labels]
    pred_labels = [label for sent_labels in cleaned_labels for label in sent_labels]

    # Ensure consistent length
    min_length = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_length]
    pred_labels = pred_labels[:min_length]

    print("\n🔹 Evaluation Metrics for", filename)
    print(classification_report(true_labels, pred_labels, digits=4))

    # ------------------ Step 7: Print Sample NER Output ------------------
    print("\n🔹 Sample NER Output from", filename)
    for token, label in zip(test_tokens[0], cleaned_labels[0]):
        print(f"{token}: {label}")

    # ------------------ Step 8: Display Label Distribution ------------------
    label_counts = Counter(true_labels)
    print("\n🔹 Label Distribution in", filename)
    print(label_counts)

    print("\n" + "=" * 60)  # Separator for better readability
