from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Initialize the pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none")


# Full legal text input
example = """
Input Text
"""

# Step 1: Preprocess the text (normalize spacing and punctuation)
def preprocess_text(text):
    text = re.sub(r'([.,;:])', r' \1 ', text)  # Add space around punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

processed_example = preprocess_text(example)

# Step 2: Split the text into manageable chunks
def split_text(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

# Split the text with overlap
chunks = split_text(processed_example, tokenizer)

# Step 3: Process each chunk with the NER pipeline
all_results = []
for chunk in chunks:
    ner_results = nlp(chunk)
    all_results.extend(ner_results)

# Step 4: Define the manual chunking function
def manual_chunking(ner_results):
    chunks = []
    current_chunk = {"word": "", "label": None, "start": None, "end": None}

    for entity in ner_results:
        word = entity["word"]
        label = entity["entity"]
        start = entity["start"]
        end = entity["end"]

        # Check for subword tokens and clean them
        if word.startswith("##"):
            word = word[2:]  # Remove '##' and concatenate to the previous word
            current_chunk["word"] += word  # Attach subword directly
            current_chunk["end"] = end  # Update the end index
        else:
            if label.startswith("B-"):  # Start of a new entity
                if current_chunk["word"]:  # Save the previous chunk
                    chunks.append(current_chunk)
                current_chunk = {"word": word, "label": label[2:], "start": start, "end": end}  # Start a new chunk
            elif label.startswith("I-") and current_chunk["label"] == label[2:]:  # Continuation
                current_chunk["word"] += f" {word}"  # Add space before full token
                current_chunk["end"] = end  # Update the end index
            else:  # Non-entity or invalid continuation
                if current_chunk["word"]:  # Save the previous chunk
                    chunks.append(current_chunk)
                current_chunk = {"word": "", "label": None, "start": None, "end": None}

    # Add the last chunk
    if current_chunk["word"]:
        chunks.append(current_chunk)

    return chunks

# Step 5: Apply manual chunking to combined results
chunked_results = manual_chunking(all_results)

# Step 6: Print the chunked results with start and end indices
for chunk in chunked_results:
    print(f"Entity: {chunk['word']}, Label: {chunk['label']}, Start: {chunk['start']}, End: {chunk['end']}\n")
