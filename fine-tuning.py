import os
import json
import numpy as np
import datasets
import torch
from datasets import Dataset, load_dataset
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

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
def parse_iob_file(file_path):
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
eval_tokens, eval_labels = [], []
for file_path in eval_files:
    data = parse_iob_file(file_path)
    eval_tokens.extend(data["tokens"])
    eval_labels.extend(data["ner_tags"])
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
    "O": 0, "B-INS": 1, "I-INS": 2, "B-CNS": 3, "I-CNS": 4,
    "B-STA": 5, "I-STA": 6, "B-RA": 7, "I-RA": 8, "B-PROM_DATE": 9, "I-PROM_DATE": 10,
    "B-CASE_NUM": 11, "I-CASE_NUM": 12, "B-PERSON": 13, "I-PERSON": 14
}

# Only 7 entity types (excluding 'O')
ENTITY_TYPES = ['INS', 'CNS', 'STA', 'RA', 'PROM_DATE', 'CASE_NUM', 'PERSON']
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
from sklearn.metrics import confusion_matrix
import pandas as pd

ENTITY_TYPES = ['INS', 'CNS', 'STA', 'RA', 'PROM_DATE', 'CASE_NUM', 'PERSON']
ENTITY_INDEX = {name: i for i, name in enumerate(ENTITY_TYPES)}

flat_true = []
flat_pred = []
mismatch_details = []

# Reconstruct spans for analysis
for i, (label_seq, pred_seq) in enumerate(zip(labels, np.argmax(predictions, axis=2))):
    words = eval_tokens[i]
    tokenized = tokenizer(words, truncation=True, padding="max_length", max_length=512, is_split_into_words=True)
    word_ids = tokenized.word_ids()

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
                "pred": pred_entity
            })

# Create and print confusion matrix
conf_matrix = confusion_matrix(flat_true, flat_pred, labels=list(range(len(ENTITY_TYPES))))
conf_df = pd.DataFrame(conf_matrix, index=ENTITY_TYPES, columns=ENTITY_TYPES)
print("\n📊 7x7 Confusion Matrix (Rows = True, Columns = Predicted):\n")
print(conf_df)

from collections import defaultdict

# Group mismatches by true entity type
grouped_mismatches = defaultdict(list)

for entry in mismatch_details:
    grouped_mismatches[entry['true']].append(entry)

if grouped_mismatches:
    print("\n❌ Misclassified Tokens Grouped by True Entity:")
    for true_entity in sorted(grouped_mismatches.keys()):
        print(f"\n🔸 {true_entity} →")
        for item in grouped_mismatches[true_entity]:
            print(f"   • {item['text']} → {item['pred']}")
else:
    print("\n✅ No misclassifications found.")


