from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Initialize the pipeline with aggregation strategy
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'([.,;:])', r' \1 ', text)  # Add space around punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Split the text into manageable chunks
def split_text(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

# Manual chunking to handle grouped entities
def manual_chunking(ner_results):
    chunks = []
    for entity in ner_results:
        word = entity["word"]
        label = entity["entity_group"]  # Use "entity_group" instead of "entity"
        score = entity["score"] * 100  # Probability of the entity
        start = entity["start"]
        end = entity["end"]

        # Append the entity to the results
        chunks.append({"word": word, "label": label, "score": score, "start": start, "end": end})

    return chunks

# Filter entities based on context
def filter_entities(chunked_results):
    filtered_results = []
    for chunk in chunked_results:
        word = chunk["word"]
        label = chunk["label"]
        score = chunk["score"]
        filtered_results.append(chunk)
    return filtered_results

# Full workflow
example = """
The Supreme Court ruled in G.R. No. 123456 that Section 5 of Republic Act No. 6713 is constitutional. 
The decision was promulgated on March 12, 2015. According to Article 7 of the 1987 Constitution, public 
officials must uphold integrity and accountability. The case involved the Office of the Ombudsman and 
several government agencies, including the Department of Justice (DOJ) and the Commission on Audit (COA).
"""
processed_example = preprocess_text(example)
chunks = split_text(processed_example, tokenizer, overlap=50)
all_results = []
for chunk in chunks:
    ner_results = nlp(chunk)
    all_results.extend(ner_results)
chunked_results = manual_chunking(all_results)
filtered_results = filter_entities(chunked_results)

# Write final results to a text file
with open("ner_results.txt", "w", encoding="utf-8") as file:
    for chunk in filtered_results:
        file.write(f"Entity: {chunk['word']}, Label: {chunk['label']}, Probability: {chunk['score']:.2f}%, Start: {chunk['start']}, End: {chunk['end']}\n")

print("Results saved to ner_results.txt")
