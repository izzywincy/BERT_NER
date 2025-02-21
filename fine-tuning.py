# Alfred tries fine-tuning type shi fine shyt

# THIS FILE CREATES A FINE TUNED MODEL BASED ON CLEANED INPUT (fixed_data.jsonl)
# THE END OF THIS FILE ALSO TESTS THE FINE TUNED MODEL ON AN INPUTTED SENTENCE

# 📌 Step 1: Import Necessary Libraries
import os 
import json
import numpy as np
import datasets
import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)



# 📌 Step 2: Load Multiple Datasets

data_folder = "cleaned_data/"

dataset_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".jsonl")]

# Print to verify loaded files
print("✅ Found dataset files:", dataset_files)

# Load datasets
dataset = load_dataset("json", data_files={"train": dataset_files}, split="train")


# 📌 Step 3.1: Load Tokenizer 
model_name = "dslim/bert-base-NER"  # Base BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 📌 Step 3.2: Define label-to-ID mapping
LABEL_MAP = {
    "O": 0,  # Outside any entity
    "INS": 1,  # Institution
    "CNS": 2,  # Constitution
    "STA": 3,  # Statute
    "RA": 4,  # Republic Act
    "PROM_DATE": 5,  # Promulgation Date
    "CASE_NUM": 6,  # Case Number
    "PERSON": 7  # Person
}

# Reverse mapping for model output interpretation
id2label = {v: k for k, v in LABEL_MAP.items()}  # Converts 0 -> "O", 1 -> "INS", etc.
label2id = {k: v for k, v in LABEL_MAP.items()}  # Converts "INS" -> 1, etc.

# 📌 Step 3.3: Load Model with Custom Labels
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=len(LABEL_MAP), 
    id2label=id2label, 
    label2id=label2id, 
    ignore_mismatched_sizes=True
)

# 📌 Step 4: Tokenize and Align Labels
import torch

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512, return_offsets_mapping=True
    )

    labels = []
    for i, label in enumerate(examples["entities"]):
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])  # Default ignored labels

        for entity in label:
            start, end, entity_label = entity["start"], entity["end"], entity["label"]

            if entity_label not in label2id:
                print(f"⚠️ Warning: Unknown label '{entity_label}' found in dataset. Skipping.")
                continue  

            # Align labels with tokens
            for idx, (token_start, token_end) in enumerate(tokenized_inputs["offset_mapping"][i]):
                if token_start is None or token_end is None:
                    continue  # Skip special tokens
                if start <= token_start < end:  
                    label_ids[idx] = label2id[entity_label]  # Assign correct label

        labels.append(label_ids)

    tokenized_inputs.pop("offset_mapping")  # Remove offset mapping (not needed after processing)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)


for batch in tokenized_dataset:
    print(f"Input length: {len(batch['input_ids'])}, Labels length: {len(batch['labels'])}")

    if len(batch['input_ids']) != len(batch['labels']):
        raise ValueError("❌ Mismatch detected between tokenized input and label lengths!")


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

# Check label distribution in dataset
# from collections import Counter

# all_labels = [label for data in tokenized_dataset["labels"] for label in data if label != -100]
# label_counts = Counter(all_labels)

# print("Label Distribution:", {id2label[k]: v for k, v in label_counts.items()})


trainer.train()

# 📌 Step 7: Evaluate Performance
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Convert logits to class indices
    predictions = np.argmax(predictions, axis=2)

    # Remove -100 values (ignored tokens) from labels and predictions
    true_labels = []
    pred_labels = []

    for label_list, pred_list in zip(labels, predictions):
        filtered_labels = []
        filtered_preds = []

        for label, pred in zip(label_list, pred_list):
            if label != -100:  # Ignore padding tokens
                filtered_labels.append(id2label[label])
                filtered_preds.append(id2label[pred])

        if filtered_labels:  # Ensure empty lists aren't included
            true_labels.append(filtered_labels)
            pred_labels.append(filtered_preds)

    # **Check if valid labels exist before computing metrics**
    if not true_labels:
        print("❌ Error: No valid labels found for metric computation!")
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}

    # Compute metrics
    results = metric.compute(predictions=pred_labels, references=true_labels)

    return {
        "f1": results.get("overall_f1", 0.0),
        "precision": results.get("overall_precision", 0.0),
        "recall": results.get("overall_recall", 0.0),
        "accuracy": results.get("overall_accuracy", 0.0),
    }

# Run evaluation with predictions
predictions, labels, _ = trainer.predict(tokenized_dataset)  # Get predictions instead of just eval loss

# Compute Metrics
metrics = compute_metrics((predictions, labels))

# 📌 Step 8: Save and Deploy Model
model.save_pretrained("./bert-legal-ner")
tokenizer.save_pretrained("./bert-legal-ner")

# Upload to Hugging Face Hub (Optional)
# model.push_to_hub("your-username/philippines-legal-ner")
# tokenizer.push_to_hub("your-username/philippines-legal-ner")

# 📌 Step 9: Test the Fine-Tuned Model with a test set
nlp = pipeline("ner", model="./bert-legal-ner", tokenizer="./bert-legal-ner", aggregation_strategy="first")

# Input test set
text = """The Supreme Court ruled in G.R. No. 123456 that Section 5 of Republic Act No. 6713 is constitutional. 
The decision was promulgated on March 12, 2015. According to Article 7 of the 1987 Constitution, public 
officials must uphold integrity and accountability. The case involved the Office of the Ombudsman and 
several government agencies, including the Department of Justice (DOJ) and the Commission on Audit (COA)."""

results = nlp(text)


# Convert numeric labels to actual entity names
for entity in results:
    label = entity["entity_group"]  # Extract label
    if label.startswith("LABEL_"):  # If still in numeric format (e.g., "LABEL_3")
        entity["entity_group"] = id2label.get(int(label.replace("LABEL_", "")), "O")  # Map it
    # Otherwise, keep the existing label if it's already mapped


# Assuming 'results' contains the NER output
print("◆ NER Output:")
for entity in results:
    print(f"{entity['word']} : {entity['entity_group']}")
print(f"🔹 F1 Score: {metrics['f1']:.4f}")
print(f"🔹 Precision: {metrics['precision']:.4f}")
print(f"🔹 Recall: {metrics['recall']:.4f}")
