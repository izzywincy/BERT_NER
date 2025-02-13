# Alfred tries fine-tuning type shi fine shyt

# THIS FILE CREATES A FINE TUNED MODEL BASED ON CLEANED INPUT (fixed_data.jsonl)
# THE END OF THIS FILE ALSO TESTS THE FINE TUNED MODEL ON AN INPUTTED SENTENCE

# 📌 Step 1: Import Necessary Libraries
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

# 📌 Step 2: Load Dataset
dataset = load_dataset("json", data_files={"train": "fixed_data.jsonl"}, split="train")

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
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True  # Needed for token-to-char mapping
    )
    
    labels = []
    for batch_index in range(len(tokenized_inputs["input_ids"])):
        entity_labels = [-100] * len(tokenized_inputs["input_ids"][batch_index])  # Default to ignore token (-100)
        
        for ent in examples["entities"][batch_index]:  
            start, end, entity_label = ent["start"], ent["end"], ent["label"]
            entity_label_id = LABEL_MAP.get(entity_label, 0)  # Convert label to ID (default "O" -> 0)

            for i, (token_start, token_end) in enumerate(tokenized_inputs["offset_mapping"][batch_index]):
                if token_start is None or token_end is None:
                    continue  # Skip tokens with no mapping
                if token_start >= start and token_end <= end:
                    entity_labels[i] = entity_label_id  

        labels.append(entity_labels)

    # Convert to tensor and match model expectation
    tokenized_inputs["labels"] = labels
    tokenized_inputs.pop("offset_mapping")  # Remove unnecessary data
    return tokenized_inputs

# Apply to dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)


# 📌 Step 5: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bert-legal-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=5,  # Adjust based on dataset size
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

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    results = metric.compute(predictions=predictions, references=labels)
    return {
        "f1": results["overall_f1"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"]
    }

trainer.evaluate()

# 📌 Step 8: Save and Deploy Model
model.save_pretrained("./bert-legal-ner")
tokenizer.save_pretrained("./bert-legal-ner")

# Upload to Hugging Face Hub (Optional)
# model.push_to_hub("your-username/philippines-legal-ner")
# tokenizer.push_to_hub("your-username/philippines-legal-ner")

# 📌 Step 9: Test the Fine-Tuned Model with a test set
nlp = pipeline("ner", model="./bert-legal-ner", tokenizer="./bert-legal-ner", aggregation_strategy="first")

# Input test set
text = "The Supreme Court ruled on Republic Act No. 3019 on January 15, 2022."
results = nlp(text)

# Convert numeric labels to actual entity names
for entity in results:
    label = entity["entity_group"]  # Extract label
    if label.startswith("LABEL_"):  # If still in numeric format (e.g., "LABEL_3")
        entity["entity_group"] = id2label.get(int(label.replace("LABEL_", "")), "O")  # Map it
    # Otherwise, keep the existing label if it's already mapped


print("🔹 NER Output:", results)
