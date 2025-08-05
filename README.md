# Named Entity Recognition for Philippine Legal Documents Using a Fine-Tuned BERT Model
Extracting information from a large body of text can be time-intensive and laborious due to the complexity of legal data. Named Entity Recognition (NER) offers an efficient and refined approach for this task. However, existing NER models tend to face challenges when applied to the legal domain due to the complexity of the terminology and contextual dependencies.

This research proposes a customized BERT-based model tailored to identifying entities within Philippine legal documents. The study aims to improve information extraction within the domain of Philippine law, by utilizing NER. The model is fine-tuned using Supreme Court Decisions documents in the Philippine legal domain to address local entities and contexts. The fine-tuned legal NER model achieved an F1 score of 0.7092, with precision at 0.6615 and recall at 0.7644 on the post-augmented test set. This marks a significant improvement from the pre-trained baseline, which scored 0.17 in entity categorization and 0.47 in entity recognition.

Data augmentation expanded the dataset by 300%, improving performance on underrepresented classes such as RA and PROM DATE while reducing misclassifications across semantically similar entities. These results demonstrate the feasibility of adapting transformer-based NER systems to low-resource legal domains through domain-specific annotation, augmentation, and fine-tuning, offering promising applications for legal information retrieval in the Philippines.
Overview of the Deliverables’ File Structure


# Proponents and Adviser

    Proponents: Alfred Victoria, Izabella Imperial, Jose Latosa, Shanky de Gracia

    Adviser: Doc. Franchesca Laguna

    Contact: alfred_victoria@dlsu.edu.ph, izabella_imperial@dlsu.edu.ph, jose_romulo_latosa@dlsu.edu.ph, shanky_degracia@dlsu.edu.ph


# File Directory
.
├── .venv                     # Virtual environment
├── .vs                        # Visual Studio Code configuration
├── error_logs                 # Logs from error debugging
├── plots                      # Plots from model evaluation
├── queue                      # Holds the queue for pre-processed data
├── raw_data                   # Contains raw datasets for training
│   ├── [new] 5 datasets.json  # New raw dataset
│   ├── 052125.json            # Another dataset
├── train_data                 # Folder for training and fine-tuning scripts
│   ├── .gitignore
│   ├── BERT_NER.py            # Main training script for BERT
│   ├── cleaning-data.py       # Script for data cleaning
│   ├── count.py               # Script for tallying dataset
│   ├── data-augmentation.py  # Data augmentation script
│   ├── fine-tuning.py        # Script for fine-tuning BERT
│   ├── make_heatmap.py        # Generates heatmaps
│   ├── ner_results.txt        # File storing NER results
│   ├── remove_augmented.py    # Removes augmented data
│   ├── remove_cns_labels.py   # Removes CNS labels
│   ├── split_balance.py       # Splits dataset for balancing
│   └── test-model.py          # Testing the trained model
├── README.md                 # This file

Key Files:

    fine-tuning.py: Fine-tunes the BERT model on the annotated legal dataset to improve its performance in identifying entities within Philippine legal texts.

    data-augmentation.py: Implements data augmentation techniques (e.g., entity swapping) to expand the training data, especially for underrepresented entity types.

    cleaning-data.py: Cleans and preprocesses the raw data before training, ensuring proper token-label alignment and data consistency.

    test-model.py: Evaluates the trained NER model on the test dataset and generates performance metrics

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

# Data Augmentation and Splitting
1. Run `split_balance.py` to split the cleaned, `queue`, to train, eval, and test splits
- 90% of CNS goes to train, 10% goes to eval
2. Run `data-augmentation.py` to augment training data (entity swapping) in `train_data/train`
- Augment CNS in train  
3. Run command `cp train_data/train/*_aug*.iob queue/` to copy all augmented data to `queue`
4. Start training