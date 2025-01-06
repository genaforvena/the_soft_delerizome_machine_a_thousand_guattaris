import json
from transformers import AutoTokenizer

# Configuration
INPUT_FILE = "finnegans_30.txt"  # Path to your Finnegans Wake text file
MODEL_NAME = "unsloth/Llama-3.2-1B"

OUTPUT_FILE = "finnegans_wake_dataset_2.jsonl"  # Local file to save the dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the text
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the text
tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)

# Define chunk size and sliding window
CHUNK_SIZE = 2048  # Adjust based on your hardware capabilities
SLIDE_WINDOW = 512  # Overlap between chunks

# Create chunks with sliding window
chunks = [tokens[i:i + CHUNK_SIZE] for i in range(0, len(tokens), SLIDE_WINDOW)]

# Prepare dataset
dataset = []
for chunk in chunks:
    chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
    
    # Use a sliding window approach for context, instruction, and response
    context = chunk_text[:len(chunk_text)//3]
    instruction = chunk_text[len(chunk_text)//3:2*len(chunk_text)//3]
    response = chunk_text[2*len(chunk_text)//3:]
    
    dataset.append({
        "context": context,
        "instruction": instruction,
        "response": response,
    })

# Save dataset locally as a .jsonl file
with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for item in dataset:
        json.dump(item, file)
        file.write("\n")

print(f"Dataset saved locally to {OUTPUT_FILE}")
