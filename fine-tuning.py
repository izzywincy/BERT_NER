import os
import json
import numpy as np
import datasets
import torch
from datasets import Dataset, load_dataset
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###################################################################################################################################
#UPDATED
# 📌 Step 1: Load IOB Files
# Separate folders for training and evaluation
train_folder = "train_data/train"
eval_folder = "train_data/eval"

train_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith(".iob")]
eval_files = [os.path.join(eval_folder, f) for f in os.listdir(eval_folder) if f.endswith(".iob")]

file_count = len(train_files)  # Only count training files
###################################################################################################################################
def parse_iob_file(file_path, return_sources=False):
    tokens, labels = [], []
    sources = []
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
                if return_sources:
                    sources.append(os.path.basename(file_path))
                token_list, label_list = [], []
        if token_list:
            tokens.append(token_list)
            labels.append(label_list)
            if return_sources:
                sources.append(os.path.basename(file_path))
    if return_sources:
        return {"tokens": tokens, "ner_tags": labels, "sources": sources}
    return {"tokens": tokens, "ner_tags": labels}


###################################################################################################################################
#UPDATED
# Parse training data
train_tokens, train_labels = [], []
for file_path in train_files:
    data = parse_iob_file(file_path)
    train_tokens.extend(data["tokens"])
    train_labels.extend(data["ner_tags"])

# Parse evaluation data
eval_tokens, eval_labels, eval_sources = [], [], []
for file_path in eval_files:
    data = parse_iob_file(file_path, return_sources=True)
    eval_tokens.extend(data["tokens"])
    eval_labels.extend(data["ner_tags"])
    eval_sources.extend(data["sources"])

###################################################################################################################################

# ✅ Step 4: Train-Test Split
#train_texts, val_texts, train_labels, val_labels = train_test_split(
#    all_tokens, all_labels, test_size=0.2, random_state=42
#)

###################################################################################################################################
#UPDATED
train_dataset = Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels})
val_dataset = Dataset.from_dict({"tokens": eval_tokens, "ner_tags": eval_labels})
###################################################################################################################################
# 📌 Step 5: Load Tokenizer and Model

model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)

LABEL_MAP = {
    "O": 0, 
    "B-INS": 1, "I-INS": 2, 
    "B-STA": 3, "I-STA": 4, 
    "B-RA": 5, "I-RA": 6, 
    "B-PROM_DATE": 7, "I-PROM_DATE": 8,
    "B-CASE_NUM": 9, "I-CASE_NUM": 10, "B-PERSON": 11, "I-PERSON": 12
}

# Only 7 entity types (excluding 'O')
ENTITY_TYPES = ['INS', 'STA', 'RA', 'PROM_DATE', 'CASE_NUM', 'PERSON']
ENTITY_INDEX = {name: i for i, name in enumerate(ENTITY_TYPES)}

id2label = {v: k for k, v in LABEL_MAP.items()}
label2id = {k: v for k, v in LABEL_MAP.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(LABEL_MAP), id2label=id2label, label2id=label2id,
    ignore_mismatched_sizes=True
)

# 📌 Step 6: Tokenization & Label Alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, padding="max_length", max_length=512, is_split_into_words=True
    )
    labels = []
    ##############
    model.eval()
    ##############
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words
        label_ids = [-100] * len(word_ids)  # Default ignored labels

        previous_word_idx = None
        for j, word_idx in enumerate(word_ids):
            if word_idx is None:  # Ignore special tokens ([CLS], [SEP])
                continue
            if word_idx != previous_word_idx:  # First token of a word
                label_ids[j] = label2id[label[word_idx]]
            else:  # Apply "I-ENTITY" to all subwords
                if label[word_idx].startswith("B-"):
                    label_ids[j] = label2id[label[word_idx].replace("B-", "I-")]
                else:
                    label_ids[j] = label2id[label[word_idx]]
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
# Debug
print("\n📝 Sentence:", " ".join(train_tokens[0]))
tokenized_output = tokenizer(train_tokens[0], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_output["input_ids"])
word_ids = tokenized_output.word_ids()
print("Tokens:", tokens)
print("Word IDs:", word_ids)


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# 📌 Step 7: Training Arguments

training_args = TrainingArguments(
    output_dir="./bert-legal-ner",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,  # Increase to 15-20
    learning_rate=3e-5,  # Slight increase on learning rate
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="epoch",
)



# 📌 Step 8: Train Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()

# 📌 Step 9: Evaluate Model
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    true_labels, pred_labels = [], []
    for label_list, pred_list in zip(labels, predictions):
        filtered_labels, filtered_preds = [], []
        for label, pred in zip(label_list, pred_list):
            if label != -100:
                filtered_labels.append(id2label[label])
                filtered_preds.append(id2label[pred])
        if filtered_labels:
            true_labels.append(filtered_labels)
            pred_labels.append(filtered_preds)
    return metric.compute(predictions=pred_labels, references=true_labels)

predictions, labels, _ = trainer.predict(val_dataset)
metrics = compute_metrics((predictions, labels))
print(f"Documents trained: {file_count}")
print(f"🔹 F1 Score: {metrics['overall_f1']:.4f}")
print(f"🔹 Precision: {metrics['overall_precision']:.4f}")
print(f"🔹 Recall: {metrics['overall_recall']:.4f}")

# 📌 Step 10: Save Model
model.save_pretrained("./bert-legal-ner")
tokenizer.save_pretrained("./bert-legal-ner")

# 📌 Step 11: Matrix
# 📌 Step 11: Matrix
from sklearn.metrics import confusion_matrix
import pandas as pd

ENTITY_TYPES = ['INS', 'STA', 'RA', 'PROM_DATE', 'CASE_NUM', 'PERSON']
ENTITY_INDEX = {name: i for i, name in enumerate(ENTITY_TYPES)}

flat_true = []
flat_pred = []
mismatch_details = []

# Reconstruct spans for analysis
for i, (label_seq, pred_seq) in enumerate(zip(labels, np.argmax(predictions, axis=2))):
    words = eval_tokens[i]
    
    tokenized = tokenizer(
    words,
    truncation=True,
    padding="max_length",
    max_length=512,
    is_split_into_words=True,
    return_tensors="pt"
    )
    
    word_ids = tokenized.word_ids()
    source_file = eval_sources[i]

    for true_id, pred_id, word_id in zip(label_seq, pred_seq, word_ids):
        if true_id == -100 or word_id is None:
            continue

        token_text = words[word_id]
        true_tag = id2label[true_id]
        pred_tag = id2label[pred_id]

        true_entity = true_tag.replace("B-", "").replace("I-", "")
        pred_entity = pred_tag.replace("B-", "").replace("I-", "")

        # Skip if both are "O" or not in ENTITY_TYPES
        if true_tag == "O" and pred_tag == "O":
            continue
        if true_entity not in ENTITY_INDEX or pred_entity not in ENTITY_INDEX:
            continue

        flat_true.append(ENTITY_INDEX[true_entity])
        flat_pred.append(ENTITY_INDEX[pred_entity])

        if true_entity != pred_entity:
            mismatch_details.append({
                "text": token_text,
                "true": true_entity,
                "pred": pred_entity,
                "file": source_file
            })

# Create and print confusion matrix
conf_matrix = confusion_matrix(flat_true, flat_pred, labels=list(range(len(ENTITY_TYPES))))
conf_df = pd.DataFrame(conf_matrix, index=ENTITY_TYPES, columns=ENTITY_TYPES)

missed_counts = []
for i, entity in enumerate(ENTITY_TYPES):
    total_true = conf_matrix[i, :].sum()      # Total times the entity appears in ground truth
    correct = conf_matrix[i, i]               # Correct predictions
    missed = total_true - correct             # Missed = total - correct
    missed_counts.append(missed)

conf_df['Missed'] = missed_counts  # This adds the column to the far right

# Print updated confusion matrix
print("\n📊 7x7 Confusion Matrix with 'Missed' Column (Rows = True, Columns = Predicted):\n")
print(conf_df)

# Flatten and decode predicted and true labels
true_labels_flat = []
pred_labels_flat = []

label_map = id2label  # from earlier

for label_seq, pred_seq in zip(labels, np.argmax(predictions, axis=2)):
    for true_id, pred_id in zip(label_seq, pred_seq):
        if true_id != -100:
            true_labels_flat.append(label_map[true_id])
            pred_labels_flat.append(label_map[pred_id])

print("\n🔹 Overall Classification Report:")
print(classification_report(true_labels_flat, pred_labels_flat, digits=4, zero_division=0))

from collections import defaultdict

from collections import defaultdict

seen_mismatches = set()
grouped_mismatches = defaultdict(list)

for i, (label_seq, pred_seq) in enumerate(zip(labels, np.argmax(predictions, axis=2))):
    words = eval_tokens[i]
    source_file = eval_sources[i]

    tokenized = tokenizer(
        words,
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt"
    )

    word_ids = tokenized.word_ids()

    for true_id, pred_id, word_id in zip(label_seq, pred_seq, word_ids):
        if true_id == -100 or word_id is None:
            continue

        token = words[word_id]
        true_tag = id2label[true_id]
        pred_tag = id2label[pred_id]

        # Skip both "O"
        if true_tag == "O" and pred_tag == "O":
            continue

        # Skip if entity types are the same (B-/I- variants treated as correct)
        if true_tag != "O" and pred_tag != "O":
            if true_tag.replace("B-", "").replace("I-", "") == pred_tag.replace("B-", "").replace("I-", ""):
                continue

        true_entity = true_tag.replace("B-", "").replace("I-", "")
        pred_entity = pred_tag.replace("B-", "").replace("I-", "")

        # Skip if true label is "O" or if both entities are untracked
        if true_tag == "O":
            continue
        if true_entity not in ENTITY_INDEX and pred_entity not in ENTITY_INDEX:
            continue

        key = (true_entity, pred_entity, token, source_file)
        if key not in seen_mismatches:
            seen_mismatches.add(key)
            grouped_mismatches[true_entity].append((pred_entity, token, source_file))

# Final printout
if grouped_mismatches:
    print("\n❌ Unique Misclassified Tokens Grouped by True Entity:")
    for true_entity in sorted(grouped_mismatches):
        print(f"\n🔸 {true_entity} →")
        for pred_entity, token, source_file in grouped_mismatches[true_entity]:
            print(f"   • '{token}' → predicted as {pred_entity} (from {source_file})")
else:
    print("\n✅ No misclassifications found.")

# Print total counts of true entity labels
print("Total true entity labels counted:", len(flat_true))
print("Breakdown:", np.bincount(flat_true))

# 📌 Step 12: Construct NER-style 2x2 Confusion Matrix
tp = 0  # Correctly predicted entity with correct type
fp = 0  # Predicted as entity but wrong type or shouldn't be an entity
fn = 0  # Should be entity but model missed it
tn = 0  # Correctly predicted non-entity

for i, (label_seq, pred_seq) in enumerate(zip(labels, np.argmax(predictions, axis=2))):
    words = eval_tokens[i]
    tokenized = tokenizer(
        words,
        truncation=True,
        padding="max_length",
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt"
    )
    word_ids = tokenized.word_ids()

    for true_id, pred_id, word_id in zip(label_seq, pred_seq, word_ids):
        if word_id is None or true_id == -100:
            continue

        true_tag = id2label[true_id]
        pred_tag = id2label[pred_id]

        if true_tag == "O" and pred_tag == "O":
            tn += 1
        elif true_tag == "O" and pred_tag != "O":
            fp += 1
        elif true_tag != "O" and pred_tag == "O":
            fn += 1
        elif true_tag == pred_tag:
            tp += 1
        else:
            fp += 1  # wrong entity type

# Display matrix
matrix_2x2 = pd.DataFrame(
    [[tp, fn],
     [fp, tn]],
    index=["Actual: Entity", "Actual: Non-Entity"],
    columns=["Predicted: Entity", "Predicted: Non-Entity"]
)

print("\n🧮 2x2 NER-Specific Confusion Matrix:\n")
print(matrix_2x2)
