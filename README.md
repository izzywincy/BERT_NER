# BERT_NER

This project fine-tunes a BERT-based Named Entity Recognition (NER) model to extract key legal entities from Philippine legal texts. The model is trained to identify institutions, laws, case numbers, promulgation dates, and other legal references.

# Named Entity Categories
The model recognizes the following entities:

| Label |    Entity Type |
| ----- | ------------- |
| INS    | Institution (e.g., Supreme Court, DOJ) |
| CNS    | Constitution (e.g., 1987 Constitution, Article 7) |
| STA    | Statute (e.g., Section 5, Article 10) |
| RA    | Republic Act (e.g., Republic Act No. 6713) |
| PROM_DATE    | Promulgation Date (e.g., March 12, 2015) |
| CASE_NUM |    Case Number (e.g., G.R. No. 123456) |
| PERSON |    Person (e.g., Juan Dela Cruz) |

# Setup
pip install transformers torch numpy json datasets

# Training 
1. Load and preprocess the dataset
- Automatically reads .jsonl files from cleaned_data/
- Ensures proper token-label alignment
2. Fine-tune BERT with token classification
3. Evaluate the model using seqeval
4. Save and deploy the trained model

# Running
python fine-tuning.py
