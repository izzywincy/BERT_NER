from transformers import BertForTokenClassification, AutoTokenizer
import torch
import os
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from collections import Counter

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

test_folder = "./train_data/test"
test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".iob")]

id2label = model.config.id2label

# ------------------ Step 3: Accumulate Predictions ------------------
all_true_labels = []
all_pred_labels = []
all_tokens = []
all_sources = []

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

for file_path in test_files:
    test_tokens, test_labels = load_iob_file(file_path)
    source_tags = [os.path.basename(file_path)] * len(test_tokens)

    tokenized_inputs = tokenizer(
        test_tokens, padding=True, truncation=True, is_split_into_words=True, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**tokenized_inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    cleaned_labels = align_predictions(predictions, tokenized_inputs)
    # Flatten token list for token tracking
    file_tokens = [token for sent_tokens in test_tokens for token in sent_tokens]
    file_sources = [src for sent, src in zip(test_tokens, source_tags) for _ in sent]


    file_true = [label for sent_labels in test_labels for label in sent_labels]
    file_pred = [label for sent_labels in cleaned_labels for label in sent_labels]

    min_length = min(len(file_true), len(file_pred), len(file_tokens), len(file_sources))
    file_true = file_true[:min_length]
    file_pred = file_pred[:min_length]
    file_tokens = file_tokens[:min_length]
    file_sources = file_sources[:min_length]

    all_true_labels.extend(file_true)
    all_pred_labels.extend(file_pred)
    all_tokens.extend(file_tokens)
    all_sources.extend(file_sources)


# ------------------ Step 4: Global Evaluation ------------------
print("\n========================= 📊 GLOBAL EVALUATION =========================")

test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".iob")]

from evaluate import load
seqeval = load("seqeval")

# Re-group flat lists into sentence-level lists for seqeval
sentence_preds = []
sentence_trues = []

current_pred = []
current_true = []

for i in range(len(all_tokens)):
    current_pred.append(all_pred_labels[i])
    current_true.append(all_true_labels[i])

    # End of sentence assumed if next token belongs to new file or original sentence structure
    if i == len(all_tokens) - 1 or all_sources[i] != all_sources[i + 1]:
        sentence_preds.append(current_pred)
        sentence_trues.append(current_true)
        current_pred = []
        current_true = []

results = seqeval.compute(predictions=sentence_preds, references=sentence_trues)
print(f"\n📄 Documents tested: {len(test_files)}")
print(f"\n🔹 F1 Score: {results['overall_f1']:.4f}")
print(f"🔹 Precision: {results['overall_precision']:.4f}")
print(f"🔹 Recall: {results['overall_recall']:.4f}")


# 🔹 Classification Report
print("\n🔹 Overall Classification Report:")
print(classification_report(all_true_labels, all_pred_labels, digits=4))

# 🔹 7×7 Entity-Level Confusion Matrix
ENTITY_TYPES = ['INS', 'CNS', 'STA', 'RA', 'PROM_DATE', 'CASE_NUM', 'PERSON']
ENTITY_INDEX = {e: i for i, e in enumerate(ENTITY_TYPES)}

def strip_prefix(label):
    return label.replace("B-", "").replace("I-", "")

flat_true, flat_pred = [], []
for t, p in zip(all_true_labels, all_pred_labels):
    if t != "O" and p != "O":
        t_clean = strip_prefix(t)
        p_clean = strip_prefix(p)
        if t_clean in ENTITY_INDEX and p_clean in ENTITY_INDEX:
            flat_true.append(ENTITY_INDEX[t_clean])
            flat_pred.append(ENTITY_INDEX[p_clean])

conf_matrix = confusion_matrix(flat_true, flat_pred, labels=list(range(len(ENTITY_TYPES))))
conf_df = pd.DataFrame(conf_matrix, index=ENTITY_TYPES, columns=ENTITY_TYPES)

missed_counts = []
for i, entity in enumerate(ENTITY_TYPES):
    total_true = conf_matrix[i, :].sum()
    correct = conf_matrix[i, i]
    missed = total_true - correct
    missed_counts.append(missed)
conf_df["Missed"] = missed_counts

print("\n📊 Global Confusion Matrix (Entity-Level):")
print(conf_df)

# 🔹 2×2 NER-Style Confusion Matrix
tp = fp = fn = tn = 0
for true, pred in zip(all_true_labels, all_pred_labels):
    if true == "O" and pred == "O":
        tn += 1
    elif true == "O" and pred != "O":
        fp += 1
    elif true != "O" and pred == "O":
        fn += 1
    elif true == pred:
        tp += 1
    else:
        fp += 1  # Wrong entity type still counts as FP

matrix_2x2 = pd.DataFrame(
    [[tp, fn],
     [fp, tn]],
    index=["Actual: Entity", "Actual: Non-Entity"],
    columns=["Predicted: Entity", "Predicted: Non-Entity"]
)

print("\n🧮 Global 2×2 NER-Style Confusion Matrix:")
print(matrix_2x2)

from collections import defaultdict

seen_mismatches = set()
grouped_mismatches = defaultdict(list)

for true, pred, token, source in zip(all_true_labels, all_pred_labels, all_tokens, all_sources):
    if true == "O" and pred == "O":
        continue

    if strip_prefix(true) == strip_prefix(pred):
        continue  # skip if entity types are the same


    true_entity = strip_prefix(true)
    pred_entity = strip_prefix(pred)

    if true == "O":
        continue

    if true_entity not in ENTITY_INDEX and pred_entity not in ENTITY_INDEX:
        continue

    if true != pred:
        key = (true_entity, pred_entity, token, source)
        if key not in seen_mismatches:
            seen_mismatches.add(key)
            grouped_mismatches[true_entity].append((pred_entity, token, source))

# Final printout
if grouped_mismatches:
    print("\n❌ Unique Misclassified Tokens Grouped by True Entity:")
    for true_entity in sorted(grouped_mismatches):
        print(f"\n🔸 {true_entity} →")
        for pred_entity, token, source in grouped_mismatches[true_entity]:
            print(f"   • '{token}' → predicted as {pred_entity} (from {source})")
else:
    print("\n✅ No misclassifications found.")
