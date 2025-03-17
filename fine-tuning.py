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
file_count = len(dataset_files)
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
print("\n📝 Sentence:", " ".join(train_texts[0]))
tokenized_output = tokenizer(train_texts[0], is_split_into_words=True)
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
