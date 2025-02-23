# Alfred tries fine-tuning type shi fine shyt
# THIS FILE CREATES A FINE TUNED MODEL BASED ON CLEANED INPUT (IOB files)
# THE END OF THIS FILE ALSO TESTS THE FINE TUNED MODEL ON AN INPUTTED SENTENCE

# 📌 Step 1: Import Necessary Libraries
import os
import json
import numpy as np
import datasets
import torch
from datasets import Dataset, load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

# 📌 Step 2: Load Multiple IOB Files from cleaned_data Folder
data_folder = "cleaned_data/"
dataset_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".iob")]
print("✅ Found dataset files:", dataset_files)

# Function to parse IOB files into a dataset format
def parse_iob_file(file_path):
    tokens = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        token_list = []
        label_list = []
        for line in f:
            line = line.strip()
            if line:  # Non-blank line
                token, label = line.split("\t")
                token_list.append(token)
                label_list.append(label)
            else:  # Blank line indicates end of a sentence
                tokens.append(token_list)
                labels.append(label_list)
                token_list = []
                label_list = []
        if token_list:  # Add any remaining tokens/labels
            tokens.append(token_list)
            labels.append(label_list)
    return {"tokens": tokens, "ner_tags": labels}

# Parse all IOB files into a single dataset
all_tokens = []
all_labels = []
for file_path in dataset_files:
    data = parse_iob_file(file_path)
    all_tokens.extend(data["tokens"])
    all_labels.extend(data["ner_tags"])

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({"tokens": all_tokens, "ner_tags": all_labels})
print(f"✅ Loaded {len(dataset)} sentences from IOB files.")

# 📌 Step 3.1: Load Tokenizer
model_name = "dslim/bert-base-NER"  # Base BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 📌 Step 3.2: Define Label-to-ID Mapping
LABEL_MAP = {
    "O": 0,  # Outside any entity
    "B-INS": 1,  # Institution (beginning)
    "I-INS": 2,  # Institution (inside)
    "B-CNS": 3,  # Constitution (beginning)
    "I-CNS": 4,  # Constitution (inside)
    "B-STA": 5,  # Statute (beginning)
    "I-STA": 6,  # Statute (inside)
    "B-RA": 7,  # Republic Act (beginning)
    "I-RA": 8,  # Republic Act (inside)
    "B-PROM_DATE": 9,  # Promulgation Date (beginning)
    "I-PROM_DATE": 10,  # Promulgation Date (inside)
    "B-CASE_NUM": 11,  # Case Number (beginning)
    "I-CASE_NUM": 12,  # Case Number (inside)
    "B-PERSON": 13,  # Person (beginning)
    "I-PERSON": 14,  # Person (inside)
}
id2label = {v: k for k, v in LABEL_MAP.items()}  # Converts 0 -> "O", 1 -> "B-INS", etc.
label2id = {k: v for k, v in LABEL_MAP.items()}  # Converts "B-INS" -> 1, etc.

# 📌 Step 3.3: Load Model with Custom Labels
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(LABEL_MAP),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# 📌 Step 4: Tokenize and Align Labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, padding="max_length", max_length=512, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their corresponding word
        label_ids = [-100] * len(word_ids)  # Default ignored labels
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:  # Special tokens
                continue
            if word_idx != previous_word_idx:  # Only label the first token of a word
                label_ids[word_idx] = label2id[label[word_idx]]
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 📌 Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bert-legal-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=1,  # Adjust based on dataset size
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none"
)

# 📌 Step 6: Fine-Tune the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Change if you have a separate test dataset
    tokenizer=tokenizer
)

trainer.train()

# 📌 Step 7: Evaluate Performance
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []
    for label_list, pred_list in zip(labels, predictions):
        filtered_labels = []
        filtered_preds = []
        for label, pred in zip(label_list, pred_list):
            if label != -100:  # Ignore padding tokens
                filtered_labels.append(id2label[label])
                filtered_preds.append(id2label[pred])
        if filtered_labels:
            true_labels.append(filtered_labels)
            pred_labels.append(filtered_preds)
    if not true_labels:
        print("❌ Error: No valid labels found for metric computation!")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    results = metric.compute(predictions=pred_labels, references=true_labels)
    return {
        "f1": results.get("overall_f1", 0.0),
        "precision": results.get("overall_precision", 0.0),
        "recall": results.get("overall_recall", 0.0),
        "accuracy": results.get("overall_accuracy", 0.0),
    }

predictions, labels, _ = trainer.predict(tokenized_dataset)
metrics = compute_metrics((predictions, labels))

# 📌 Step 8: Save and Deploy Model
model.save_pretrained("./bert-legal-ner")
tokenizer.save_pretrained("./bert-legal-ner")

# 📌 Step 9: Test the Fine-Tuned Model with a Test Set
nlp = pipeline("ner", model="./bert-legal-ner", tokenizer="./bert-legal-ner", aggregation_strategy="first")

# Input test set
text = """
    Before the Court is a petition for review on certiorari1 assailing the Amended Decision2 dated January 8, 2010 and the Resolution3 dated August 3, 2010 of the Court of Appeals (CA) in CA-G.R. CV No. 82888, which: (a) reversed and set aside its earlier Decision4 dated July 6, 2009, dismissing Land Registration (LRC) Case No. TG-898 without prejudice; and (b) affirmed the Decision5 dated April 1, 2003 of the Regional Trial Court of Tagaytay City, Branch 18 (RTC), approving respondent Banal na Pag-aaral, Phil., Inc.'s (respondent) application for registration.
"""
results = nlp(text)

# Convert numeric labels to actual entity names
for entity in results:
    label = entity["entity_group"]  # Extract label
    if label.startswith("LABEL_"):  # If still in numeric format (e.g., "LABEL_3")
        entity["entity_group"] = id2label.get(int(label.replace("LABEL_", "")), "O")  # Map it

# Print NER Output
print("◆ NER Output:")
for entity in results:
    print(f"{entity['word']} : {entity['entity_group']}")
print(f"🔹 F1 Score: {metrics['f1']:.4f}")
print(f"🔹 Precision: {metrics['precision']:.4f}")
print(f"🔹 Recall: {metrics['recall']:.4f}")