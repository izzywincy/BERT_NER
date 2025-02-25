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

# debug: check if dataset is balanced
# 
# from collections import Counter
# 
# # Flatten all labels to count their distribution
# flat_labels = [label for sublist in all_labels for label in sublist]
# label_counts = Counter(flat_labels)
# 
# # Print label distribution
# print("\n🔍 Training Label Distribution:")
# for label, count in label_counts.items():
#     print(f"{label}: {count}")
# 
# # Check if "O" is overwhelmingly dominant
# if label_counts.get("O", 0) > 0.95 * sum(label_counts.values()):
#     print("⚠️ WARNING: More than 95% of labels are 'O'. The model may not learn named entities properly.")
#     #Fix:
#     # Increase dataset size (more annotated data).
#     # Oversample underrepresented labels by adding duplicate samples of "B-RA", "B-INS", etc.
#     # Reduce "O" labels if possible.
#     
# Comment: Most are O.


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
    # debug: check if tokens are misaligned
    # for i, word in enumerate(examples["tokens"]):
    #     print(f"🔹 Token: {word} → Label: {examples['ner_tags'][i]}")
    # Comment: Aligned

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
    evaluation_strategy="steps",  # Evaluate more frequently
    eval_steps=5,  # Adjust based on dataset size
    save_strategy="steps",  # Save more frequently
    save_steps=5,  # Save more often due to small dataset
    logging_dir="./logs",
    num_train_epochs=30,  # Increase since dataset is small
    per_device_train_batch_size=2,  # Reduce batch size for better updates
    per_device_eval_batch_size=2,
    warmup_steps=0,  # Reduce significantly
    weight_decay=0.01,
    logging_steps=2,  # More frequent logging
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

#TODO: Metrics might be inaccurate
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
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

# Input test set
#text = """
#    Before the Court is a petition for review on certiorari assailing the Amended Decision dated January 8, 2010 and the Resolution dated August 3, 2010 of the Court of Appeals (CA) in CA-G.R. CV No. 82888, which: (a) reversed and set aside its earlier Decision dated July 6, 2009, dismissing Land Registration (LRC) Case No. TG-898 without prejudice; and (b) affirmed the Decision dated April 1, 2003 of the Regional Trial Court of Tagaytay City, Branch 18 (RTC), approving respondent Banal na Pag-aaral, Phil., Inc.'s (respondent) application for registration.
#"""
#test_text = "The Supreme Court ruled under Republic Act No. 3019."
test_text = """
Manila
EN BANC
[ G.R. No. 261280, October 03, 2023 ]
INCUMBENT AND FORMER EMPLOYEES OF THE NATIONAL ECONOMIC AND DEVELOPMENT AUTHORITY (NEDA) REGIONAL OFFICE (RO) XIII: DELA CALZADA, MICHELLE P., CARDONA-ATO, ELVIE, BATINCILA, GLENN B., BERIDO, JAZMIN D., CADUYAC, FIDES JOY A., CALAMBA, MARY JEAN G., CARIÑO, MYLAH FAYE AURORA B., CASTILLON, MITCHELL C., GIDACAN, EMMANUEL Z., HARTING, GRAZIELLA C., JAQUILMAC, ANN B., LARIBA, ELSIE E., MAUR, MELANIE A., MENDEZ, RHEA MAE C., MICULOB, IAN G., MISSION, NAOMI T., NEISLER, ANNA LORAINE M., OLAM, GEMIMA A., PARADIANG, EDDIE B. TENER, RENANTE O.; TORRALBA, APRIL KRISTINE G., TOROTORO, SHIRLEY C., VERDUN, SHERWIN E., AND VILLANUEVA, FRANCISCO ROMULUS C., VS. COMMISSION ON AUDIT, CHAIRPERSON MICHAEL G. AGUINALDO, COMMISSIONER ROLANDO C. PONDOC, THE REGIONAL DIRECTOR COA REGIONAL OFFICE CARAGA THE CLUSTER DIRECTOR, CLUSTER 2  LEGISLATIVE AND OVERSIGHT, NATIONAL GOVERNMENT SECTOR, RESPONDENTS.
D E C I S I O N
LOPEZ, M., J.:
May the Commission on Audit (COA) Proper review and reverse its ruling of exemption motu proprio ? This is the primordial issue raised in this Petition for Certiorari under Rule 64, in relation to Rule 65, of the Rules of Court, assailing the December 22, 2021 Resolution of the COA Proper in Decision No. 2021-491.
Facts
Pursuant to Civil Service Commission (CSC) Memorandum Circular (MC) No. 01, which laid down the revised policies on Program on Awards and Incentives for Service Excellence, the National Economic and Development Authority (NEDA) issued Office Circular (OC) No. 03-2005 providing the guidelines for the establishment and implementation of its own award system, known as the NEDA Awards and Incentives System (NAIS). Among the awards available under the NAIS is the Cost Economy Measure Award (CEMA)a monetary incentive "[g]ranted to an employee or team whose contributions such as ideas, suggestions, inventions, discoveries or performance of functions result in savings in terms of manhours and cost or otherwise benefit the agency and government as a whole."
NEDA Regional Office XIU-Caraga Region (NEDA Caraga) then granted across-the-board CEMA to its employees in 2010, 2011, and 2012, charged against the agency's year-end savings.
However, upon post-audit, the Audit Team Leader (ATL) and Supervising Auditor (SA) found that the CEMA was irregular and unauthorized because it was: (1) not in accord with the Total Compensation Framework established under the Senate and House of Representatives Joint Resolution (JR) No. 04; (2) not supported by specific appropriations; and (3) not supported by clear and sufficient indicators, baselines, metrics, or standards that would justify entitlement to the award. Thus, the ATL and SA issued an Audit Observation Memorandum (AOM), recommending the refund of the CEMA and imposition of sanctions against the erring officials and employees. NEDA Caraga officials were also reminded to ensure compliance with the applicable laws and regulations in its future grant of monetary incentives.
Then NEDA Caraga Regional Director, Carmencita S. Cochingco (Cochingco) responded in a Letter that argued for the validity of the CEMA and invoked good faith in its grant and receipt.
Unconvinced, the ATL and SA issued an ND, disallowing the CEMA disbursed from 2010 to 2012, amounting to an aggregate of PHP 882,759., for the same reasons stated in the AOM. Conchingco, together with the other NEDA Caraga officers, namely: Venus S. Derequito (Derequito); Sherwin T. Sells (Sells); Elsie O. Casurra (Casurra); and Jayson G. Dy (Dy), were held liable under the ND for approving and/or certifying the grant of CEMA. NEDA Caraga employees, which include Michelle P. Dela Calzada, et al. (petitioners), were also made liable under the ND as recipients. The ND was sent to NEDA Caraga through Cochingco and its accountant, Dy.
Insisting on the validity of the CEMA grant, then NEDA Caraga Officer-in-Charge Mylah Faye Aurora B. Cariño questioned the ND before the COA National Government Sector (NGS)  Cluster 2. In its Decision No. 2015-03, the NGS affirmed the validity of the ND, as well as the liability of the approving and/or certifying officers. But as mere passive recipients of the disallowed amount, petitioners Michael P. Dela Calzada et al. (Dela Calzada et al.) were excused from the obligation to refund.
After automatic review, the COA Proper issued Decision No. 2018-306. It approved the NGS ruling in its entirety-confirmed the validity of the ND and liability of the approving and/or certifying officers, and the absolution of the passive recipients on the ground of good faith, citing Silang v. Commission on Audit.
Cochingco and the other officer liable under the ND, together with the incumbent Regional Director of NEDA Caraga, Bonifacio G. Uy (Uy), filed a Motion for Partial Reconsideration, asserting the validity of the CEMA grant and arguing for their exemption from liability on the ground of good faith. In the alternative, the officers prayed that they be made to refund only the amount that each of them received. Dela Calzada, et al., on the other hand, did not file a motion for reconsideration since they were already absolved from liability.
In the assailed Decision No. 2021-491, the COA Proper sustained the validity of the ND and the solidary liability of the officers to refund the total amount of disallowance. The COA proper went further to revisit its ruling on petitioners' liability despite non-participation of the petitioners in the motion:
As to the payees, this Commission revisits its ruling exempting them from liability in view of the ruling of the SC in the recent case of Chozas [v. Commission on Audit], where the SC held that:
The natural consequence of a finding that the allowances and benefits were illegally disbursed, is the consequent obligation on the part of all the recipients to restore said amounts to the government coffers. Such directive is in accord with Article 22 of the Civil Code. which states that, "[e]very person who through an act of performance by another, or any other means, acquires or comes into possession of something at the expense of the latter without just or legal ground, shall return the same to him." This principle of unjust enrichment applies when, "(i) a person is unjustly benefited: and (ii) such benefit is derived at the expense of or with damages to another."
This strict stance is evidence from the Court's recent pronouncements in Rotoras, James Arthur T Dubongco, Provincial Agrarian Reform Program Officer II of Department of Agrarian Reform Provincial Office-Cavite in Representation of Darpo-Cavite and All Its Officials and Employees [v.] Commission on Audit, and Department of Public Works and Highways [v.] COA, where the Court ordered the full restitution of all benefits unlawfully received by government employees. Furthermore, the Court in Rotoras stressed that the defense of good faith shall no longer work to exempt them [sic] the payees from such obligation, [...]:
. . . .
WHEREFORE, .... the Motion for Partial Reconsideration of [Cochingco], former Regional Director (RD), et al., all of the [NEDA RO XIII], represented by the incumbent RD, [Uy], is hereby DENIED . Accordingly, [COA] Decision No. 2018-306 dated March 15, 2018, which approved COA [NGS]  Cluster 2 Decision No. 2015-03 dated January 12, 2015, is AFFIRMED with MODIFICATION, in that the employees who received the benefits remain liable to refund the amount they received. (Citations omitted and emphasis supplied)
Aggrieved, Dela Calzada et al. challenge the Decision. They argue that the COA Proper committed grave abuse of discretion when it reversed its previous ruling, which had already attained finality, excluding them from liability. They emphasized that they were not parties to the officers' Motion for Partial Reconsideration, and their right to due process was violated. In any case, petitioners invoke good faith and claim that the CEMA that they received has valid and sufficient factual and legal bases.
The COA Proper, through the Office of the Solicitor General (OSG), counters that the Motion for Partial Reconsideration prevented Decision No. 2018-306 from attaining finality since the issue on the validity of the ND is not severable from the issue on the liabilities. Assuming that Decision No. 2018-306 had already attained finality, the OSG posits that the exceptions on immutability of judgments apply, i.e., existence of special or compelling circumstances. 1aшphi1 Specifically, the OSG argues that the application of the supervening jurisprudence and the correction of the inequitable effect of making the officers solely liable for the disallowed amount are sufficient justifications for the COA Proper to rectify its previous ruling on petitioners' liability.
Issue
Whether the COA Proper committed grave abuse of discretion in reinstating the liability of Dela Calzada et al. in the ND based on the Motion for Partial Reconsideration filed by the approving and/or certifying officers.
Ruling
We answer in the affirmative.
Generally, this Court sustains the rulings of the COA in deference to its constitutional mandate with regard to expenditures of government funds, as well as its presumed expertise in the laws entrusted to them to enforce. However, through the extraordinary writ of certiorari, the Court would annul decisions and resolutions of the COA when it has clearly acted without or in excess of jurisdiction, or with grave abuse of discretion amounting to lack or excess of jurisdiction." We have consistently explained:
The special civil action for certiorari is intended for correction of errors of jurisdiction ... or grave abuse of discretion amounting to lack or excess of jurisdiction. Its principal office is ... to keep the inferior court within the parameters of its jurisdiction or to prevent it from committing such a grave abuse of discretion amounting to lack or excess of jurisdiction.
. . . .
Excess of jurisdiction as distinguished from absence of jurisdiction means that an act, though within the general power of a tribunal, board or officer, is not authorized and invalid with respect to the particular proceeding, because the conditions which alone authorize the exercise of general power in respect of it are wanting. Without jurisdiction means lack or want of legal power, right or authority to hear and determine a cause or causes, considered either in general or with reference to a particular matter. It means lack of power to exercise authority. Grave abuse of discretion implies such capricious and whimsical exercise of judgment as is equivalent to lack [or excess] of jurisdiction or, in other words, where the power is exercised in an arbitrary manner by reason of passion, prejudice, or personal hostility, and it must be so patent or gross as to amount to an evasion of a positive duty or to a virtual refusal to perform the duty enjoined or to act at all in contemplation of law. (Citations omitted and emphasis supplied)
In this case, we find that the COA Proper committed grave abuse of discretion when it reviewed and reversed its previous ruling that was no longer questioned by any party since such act deviated from the applicable rules under the 2009 Revised Rules of Procedure of the [COA] (RRPC) the Rules of Court, the doctrine of immutability of final judgments, and the principle of prospective overruling. In addition, Dela Calzada et al.'s rudimentary right to procedural due process was violated. it is settled, the obstinate disregard of basic and established rule of law or procedure constitutes grave abuse of discretion.
The COA Rules of Procedure was disregarded
The COA has set its own rules to modify or revise its judgments, decisions, and resolutions. Rule X, Section 12, in relation to Rule X Sections 9 and 10 of the RRPC, as amended, requires the aggrieved party to file a motion for reconsideration for the COA Proper to review its decision or a petition for certiorari for judicial review; otherwise, the decision becomes final and executory upon the lapse of 30 days from notice. The motion, as required under Section 11 of the same Rule must be "verified and x x x [must] point out specifically the findings or conclusions of the decision which are not supported by evidence or which are contrary to law, making express reference to the testimonial or documentary evidence or the provisions of law that such finding or conclusions are alleged to be contrary to."
Although the COA Proper is authorized to motu proprio exercise its power of review, i.e., even sans appeal or motion from any party, it is only to review the decision of the COA Director which is in conflict with the Auditor's issuance. This is an internal procedure for the COA to ensure the uniformity of its officers' rulings before the decision becomes final. In all other cases of review, the COA Proper is not authorized under its rules to act motu proprio.
Here, no motion for reconsideration was filed as regards petitioners' liability since they were already absolved, but the COA Proper still reviewed and, worse, reversed its previous ruling. The COA Proper expediently assumed that Uy, as the incumbent regional director, represented the entire NEDA Caraga, including Dela Calzada et al., in the Partial Motion for Reconsideration. This assumption is erroneous because Dela Calzada et al. are not parties to the Partial Motion for Reconsideration. The caption of the motion specifically stated the names of the "movants-appellants:" Cochingco, Derequito, Sells, Casurra, and Dy, "represented by incumbent Regional Director [Uy]." Furthermore, the arguments in the motion discussed only the officers' liability. The officers prayed that they be exempt from reimbursement of the disallowed amount or, in the alternative, they be made to refund only the amount each of them received. Except for iterating petitioners' absolution, no other allegation or argument was raised as regards petitioners' liability. Finally, Dela Calzada et al. never authorized Uy to represent them in any case, unlike Cochingco, Derequito, Cassura, and Dy, who executed separate special power of attorneys expressly designating Uy as their attorney-in-fact in the Partial Motion for Reconsideration.
Clearly, from the foregoing, the COA Proper disregarded its own rules of procedure when it unilaterally reversed its ruling anent Dela Calzada et al.'s exemption from liability. As the COA Proper acted in a manner not sanctioned by the RRPC and such act affected petitioners' substantive right to property, its invalidation is warranted.
The Rules of Court allows partial reconsideration
The OSG argues that every disallowance entails a corresponding liability on every person who participated in the transactionwhether as an approving and/or certifying officer or a mere recipientand all issues arising from a disallowance must be determined jointly. In précis, the OSG maintains that the determination of the validity of the ND cannot be separated from the discussion on the rules of return of the disallowed amount. If an ND is sustained, all persons liable to settle the disallowed amount must be determined as a matter of course. Further, the solidary nature of the liability in disallowances between recipients and the approving and/or certifying officers bolsters the argument that such matters are inseparable.
The OSG is mistaken.
"[W]hen matters, issues, or claims can properly and conveniently be separately resolved," Rule 37, Section 7 of the Rules of Court permits division of judgments, i.e., a portion may be considered final and unappealable while another portion is pending appeal or reconsideration. The Rule states:
SEC. 7. Partial New Trial or Reconsideration.  If the grounds for a motion under this Rule appear to the court to affect the issues as to only a part, or less than all of the matter in controversy, or only one, or less than all, of the parties to it, the court may order a new trial or grant reconsideration as to such issues if severable without interfering with the judgment or final order upon the rest. (Emphasis supplied)
The issue on the validity of the ND is severable from the issue on petitioners' liability because whether the ND is affirmed or otherwise struck down as invalid on the officers' Partial Motion for Reconsideration, such ruling is no longer consequential upon petitioners because they had already been taken out of the picture by being absolved from any liability under the ND.
In the same vein, despite its unique solidary nature, the determination of whether an approving and/or certifying officer should be held liable in a disallowance is not dependent upon the determination of the recipients' liability because the grounds by which their liabilities arise are distinct. To elucidate, We have settled in Madera v. Commission on Audit that mere receipt of public funds without valid basis or justification gives rise to the obligation to return what was unduly received; but note that Madera also introduced jurisprudentially-recognized grounds which may excuse some or all of the recipients from such obligation. On the other hand, approving and/or certifying officers of a disallowed amount may be held civilly, administratively, and/or criminally liable upon proof that their act was tainted with bad faith, malice, or gross negligence. Clearly, the liabilities of the two sets of participants in a disallowance arise from two distinct sourcesthe recipients' liability arise from the civil law principles of unjust enrichment and solutio indebiti, while that of the officers' arise from public accountability. The two liabilities become relevant to each other only when it comes to the execution of the civil obligation to refund through the application of the concept of "net disallowed amount" as laid down in Madera, i.e., exemption of recipients from liability tempers the officers' civil liability in that the amount to be refunded shall be limited to that which remains unexcused.
Verily, contrary to the OSG's contention, the existence and extent of the officers' liability may be determined independently from the final judgment excusing petitioners from liability. Consistent with its own rules of procedure, as well as the Rules of Court, the COA Proper should have limited its resolution to the issue/s raised in the Partial Motion for Reconsideration.
Petitioners' exoneration has become final
Since no party questioned the COA Proper's affirmance of petitioners' exemption from liability, judgment on that matter undeniably lapsed into finality pursuant to Rule X, Section 9 of the RRPC, as amended. It is well­-established, a judgment becomes final and executory by operation of law. Finality becomes a fact when the reglementary period to question the judgment lapses and no such question was lodged. As a consequence, no court or tribunal (not even this Court) can review or modify a judgment that has become final. Our pronouncement in One Shipping Corp. v. Peñafiel is apropros:
In Aliviado v. Procter and Gamble Phils., Inc., [we ruled]:
It is a hornbook rule that once a judgment has become final and executory, it may no longer be modified in my respect, even if the modification is meant to correct an erroneous conclusion of fact or law, and regardless of whether the modification is attempted to be made by the court rendering it or by the highest court of the land, as what remains to be done is the purely ministerial enforcement or execution of the judgment.
The doctrine of finality of judgment is grounded on fundamental considerations of public policy and sound practice at the risk of occasional errors, the judgment of adjudicating bodies must become final and executory on some definite date fixed by law. [...], the Supreme Court reiterated that the doctrine of immutability of judgment is adhered to by necessity notwithstanding occasional errors that may result thereby, since litigations must somehow come to an end for otherwise, it would "even be more intolerable than the wrong and injustice it is designed to correct."
[Also,] [i]n Mocorro, Jr. v. Ramirez, we held that:
A definitive final judgment, however erroneous, is no longer subject to change or revision.
A decision that has acquired finality becomes immutable and unalterable. This quality of immutability precludes the modification of a final judgment, even if the modification is meant to correct erroneous conclusions of fact and law. And this postulate holds true whether the modification is made by the court that rendered it or by the highest court in the land. The orderly administration of justice requires that, at the risk of occasional errors, the judgments/resolutions of a court must reach a point of finality set by the law. The noble purpose is to write finis to dispute once and for all. This is a fundamental principle in our justice system, without which there would be no end to litigations. 1aшphi1 Utmost respect and adherence to this principle must always be maintained by those who exercise the power of adjudication. Any act, which violates such principle, must immediately be struck down. Indeed, the principle of conclusiveness of prior adjudications is not confined in its operation to the judgments of what are ordinarily known as courts, but extends to all bodies upon which judicial powers bad been conferred. (Citations omitted and emphasis supplied)
We stress, not even this Court can re-assess, much less alter, a final judgment, especially when such ruling was not challenged before the forum. As we have held in Philippine Mining Development Corp. v. Aguinaldo:
In Securities and Exchange Commission v. Commission on Audit, this Court, sitting En Banc, resolved not to rule on the merits of the civil liability of the payee-recipients who were already exonerated from liability by the COA, especially since such absolution was not questioned before this Court[.]
....
The particular circumstances (sic) is similar to [ Securities and Exchange Commission ]. To recall, the COA-CP similarly excluded the recipient employees from refunding the medical benefits they received. While they were absolved on the basis of good faith as abandoned in Madera, this Court must give due deference to the doctrine of finality of judgments, considering that their corresponding liability was no longer raised as an issue in the instant petition. Concomitantly, in [ Securities and Exchange Commission, ] the Court affirmed the COA-CP Decision, excusing the passive payees from returning the disallowed amounts on the ground of having received the same in good faith. Since their liability was no longer questioned or put in issue in the instant petition, this Court considered the COA-CP's Decision "final and immutable."
Consistently, this Court shares the observation of Senior Associate Justice Estela Perlas-Bernabe ( Justice Perlas-Bernabe ) that there is no cogent reason to deviate from the prevailing rule that when the payee­-recipients have already been finally absolved from civil liability by the COA, the merits of such absolution should be respected and not touched upon by the Court in an appeal filed by the approving or certifying officers, who in contrast, were held liable under the subject disallowances. As such, this Court maintains the absolution of herein recipient employees pursuant to the finality of judgment as elucidated in the earlier rulings of SSS and SEC. (Citations omitted and emphasis supplied)
Albeit the cited case deals with the Court's review power and not the COA Proper's motu proprio review of its unquestioned ruling as in this case, the same rationale that precludes review applies in this case, i.e., parties who do not challenge a favorable ruling for obvious reasons can no longer be prejudiced by a subsequent unilateral review. As aptly remarked by Justice Japar B. Dimaampao during deliberations, the basic tenets of fair play and due process, coupled with the severability of the issues involved, foreclose any amendment on the COA Proper's unchallenged ruling.
Jurisprudence applies prospectively
Unrelenting to its cause, the OSG posits that this case falls within the exceptions to the doctrine of immutability of judgments, citing jurisprudential developments and the need to cure the inequitable allocation of liabilities as special and compelling reasons which justify the reversal of the otherwise final ruling of exemption on petitioners' liability.
This argument is misplaced.
"[W]hen a doctrine of this Court is overruled and a different view is adopted, the new doctrine should be applied prospectively, and should not apply to parties who had relied on the old doctrine and acted on the faith thereof." With more reason, a supervening doctrine cannot justify reversal of a final judgment. This is because:
[P]ursuant to Article 8 of the Civil Code "judicial decisions applying or interpreting the laws or Constitution shall form a part of the legal system of the Philippines." But while our decisions form part of the law of the land, they are also subject to Article 4 of the Civil Code which provides that "laws shall have no retroactive effect unless the contrary is provided. This is expressed in the familiar legal maxim lex prospicit, non respicit, the law looks forward not backward. The rationale against retroactivity is easy to perceive. The retroactive application of a law usually divests rights that have already become vested or impairs the obligations of contract and hence, is unconstitutional.
Thus, it was arbitrary on the part of the COA Proper to reinstate petitioners' liability to apply the new precedentgood faith is no longer recognized as a justification to excuse recipients from liabilitywhen, at the time the previous ruling was issued and became final, prevailing jurisprudence says otherwise. To stress, Chozas v. Commission on Audit and all the other cases cited in that case, which was the basis of the COA Proper in reinstating petitioners' liability was promulgated in 2019 or after the COA Proper's previous ruling in 2018 had become final with respect to petitioners' liability. Too, it may not come amiss to note that it was only in Madera, promulgated in 2020, when the Court En Banc clarified the inapplicability of the good faith rule in excusing a recipient from liability in a disallowance. Verily, we cannot countenance a deviation from the time-honored doctrine of immutability of judgments only to violate the equally-recognized principle of prospective overruling.
Neither would the nobility of addressing the inequitable consequence of petitioners' absolution validate the reversal of the final judgment because, as it happened, it only resulted in the arbitrary reinstatement of civil liability on the part of the petitioners. In any case, at this juncture, the application of the concept of net disallowed amount as laid down in Madera may address the inequitable burden upon the officers without resorting to the reversal of the final ruling on petitioners' exemption from liability.
Petitioners' due process right was violated
The essence of procedural due process is embodied in the basic requirement of notice and a real opportunity to be heard. As we have held in Bangko Sentral ng Pilipinas v. Commission on Audit:
Due process in administrative proceedings does not require the submission of pleadings or a trial-type of hearing. [Nevertheless, it is imperative that] the party is duly notified of the allegations against him or her and is given the chance to present his or her defense. Furthermore, due process requires that the proffered defense should have been considered by the tribunal in arriving at its decision.
Here, the Court observes that all throughout the proceedings before the COA, from the auditors, the NGS, and the COA Proper, all pleadings were filed by the officers. Dela Calzada et al. were exonerated at the NGS level, which was affirmed by the COA Proper. For obvious reasons, Dela Calzada et al. no longer posed any objection and they were no longer parties before the forum. However, in the subsequent motion for reconsideration filed solely by the officers, the COA Proper applied a new doctrine and unilaterally reinstated petitioners' liability. Such act clearly violated petitioners' right to due process since they were not given the opportunity to squarely and intelligently defend themselves from such new doctrine.
In all, the COA Proper gravely abused its discretion in unilaterally reversing its final judgmentbecause of its fervor to apply a supervening case law and to address the unfair distribution of liability, it arbitrarily disregarded the established rules on its review power, causing undue prejudice to petitioners. For this reason, the Court finds it unnecessary to delve on the other arguments raised in the Petition.
ACCORDINGLY, the Petition for Certiorari is GRANTED . The Decision No. 2021-491 (Resolution) dated December 22, 2021 of the Commission on Audit Proper is SET ASIDE insofar as petitioners Michelle P. Dela Calzada et al.'s liability under Notice of Disallowance No. 2013-01(2010-2012) was reinstated. Petitioners Michelle P. Dela Calzada et al. remain EXCUSED from the civil liability to return the disallowed amount due to the finality of the Commission on Audit Proper's Decision No. 2018-306 on that aspect.
SO ORDERED.
Gesmundo, C.J., Leonen, SAJ., Caguioa, Hernando, Inting, Zalameda, Gaerlan, Rosario, J. Lopez, Marquez, Kho, Jr., and Singh, JJ., concur.
Lazaro-Javier and Dimaampao, JJ., on official business.

"""
results = nlp(test_text)

# Convert numeric labels to actual entity names
for entity in results:
    label = entity["entity_group"]  # Extract label
    if label.startswith("LABEL_"):  # If still in numeric format (e.g., "LABEL_3")
        entity["entity_group"] = id2label.get(int(label.replace("LABEL_", "")), "O")  # Map it

print("\n◆ Debugging Low Confidence Predictions:")
for entity in results:
    print(f"{entity['word']} → {entity['entity_group']} (Score: {entity['score']:.4f})")


# Print NER Output
print("◆ NER Output:")
for entity in results:
    print(f"{entity['word']} : {entity['entity_group']}")
print(f"🔹 F1 Score: {metrics['f1']:.4f}")
print(f"🔹 Precision: {metrics['precision']:.4f}")
print(f"🔹 Recall: {metrics['recall']:.4f}")

exists = any(test_text in " ".join(tokens) for tokens in all_tokens)
print(f"\n🔍 Does the test sentence exist in training data? {'✅ YES' if exists else '❌ NO'}") # Overfitting Check
