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
NOTE: It is recommended to create a Python environment to make the setup seamless.
## Python Environment Setup
### For VSCode:
1. Open the Command Palette (Crtl+Shift+P)
2. Search for the `Python: Create Environment Command`
## Dependencies Installation
Run the following command:
`pip install transformers torch numpy seqeval datasets evaluate`

# Training 
1. Load and preprocess the dataset
- Automatically reads .jsonl files from cleaned_data/
- Ensures proper token-label alignment
2. Fine-tune BERT with token classification
3. Evaluate the model using seqeval
4. Save and deploy the trained model

# Running
1. Upload annotated data (.jsonl) in /raw_data
2. Run `python cleaning-data.py` to start data clenaning
3. Run  `python fine-tuning.py` to train the cleaned dataset

# Tallying Dataset (Optional)
1. Run `count.py` to start tallying a folder
2. Provide the correct folder name
3. A tally of each document's classifications is generated as well as the folder's summary.
