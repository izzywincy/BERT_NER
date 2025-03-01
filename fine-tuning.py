import os
import json
import numpy as np
import datasets
import torch
from datasets import Dataset, load_dataset
import evaluate
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

# 📌 Step 1: Load IOB Files

data_folder = "cleaned_data/"
dataset_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".iob")]

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

# Parse all IOB files
all_tokens, all_labels = [], []
for file_path in dataset_files:
    data = parse_iob_file(file_path)
    all_tokens.extend(data["tokens"])
    all_labels.extend(data["ner_tags"])

# ✅ Step 4: Train-Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_tokens, all_labels, test_size=0.2, random_state=42
)

train_dataset = Dataset.from_dict({"tokens": train_texts, "ner_tags": train_labels})
val_dataset = Dataset.from_dict({"tokens": val_texts, "ner_tags": val_labels})

# 📌 Step 5: Load Tokenizer and Model

model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)

LABEL_MAP = {
    "O": 0, "B-INS": 1, "I-INS": 2, "B-CNS": 3, "I-CNS": 4,
    "B-STA": 5, "I-STA": 6, "B-RA": 7, "I-RA": 8, "B-PROM_DATE": 9, "I-PROM_DATE": 10,
    "B-CASE_NUM": 11, "I-CASE_NUM": 12, "B-PERSON": 13, "I-PERSON": 14
}

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
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(word_ids)
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                label_ids[word_idx] = label2id[label[word_idx]]
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# 📌 Step 7: Training Arguments

training_args = TrainingArguments(
    output_dir="./bert-legal-ner",
    evaluation_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=5,
    logging_dir="./logs",
    num_train_epochs=30,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=0,
    weight_decay=0.01,
    logging_steps=2,
    report_to="none"
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

print(f"🔹 F1 Score: {metrics['overall_f1']:.4f}")
print(f"🔹 Precision: {metrics['overall_precision']:.4f}")
print(f"🔹 Recall: {metrics['overall_recall']:.4f}")

# 📌 Step 10: Save Model
model.save_pretrained("./bert-legal-ner")
tokenizer.save_pretrained("./bert-legal-ner")
