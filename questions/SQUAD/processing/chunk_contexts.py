from pathlib import Path
import json
from typing import Literal
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

DATASET_SPLIT: Literal["train", "validation"] = "validation"
CHUNK_SIZE = 128

DATA_DIR = Path(__file__).parent.parent / 'data'
INPUT_FILE = DATA_DIR / DATASET_SPLIT / 'context_groups.json'
OUTPUT_FILE = DATA_DIR / DATASET_SPLIT / 'context_chunks.json'

nltk.download('punkt_tab')

# tokenizer of the PISCO backbone model
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2"
)

with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

output_data = {}

for i, (hash_key, context_group) in enumerate(data.items()):
    context_ids = tokenizer.encode(
        context_group["context"],
        add_special_tokens=False,
    )

    print(f"Progress: {i}/{len(data)}", end="\r")

    if len(context_ids) <= CHUNK_SIZE:
        output_data[hash_key] = {
            "context_chunks": [context_group["context"]],
            "token_lengths": [len(context_ids)],
        }
    else:
        sentences = sent_tokenize(context_group["context"])
        current_chunk = []
        current_length = 0
        context_chunks = []
        token_lengths = []

        for sentence in sentences:
            sentence_ids = tokenizer.encode(
                sentence,
                add_special_tokens=False,
            )
            sentence_length = len(sentence_ids)

            if current_length + sentence_length <= CHUNK_SIZE:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                context_chunks.append(" ".join(current_chunk))
                token_lengths.append(current_length)
                current_chunk = [sentence]
                current_length = sentence_length

        if current_chunk:
            context_chunks.append(" ".join(current_chunk))
            token_lengths.append(current_length)

        output_data[hash_key] = {
            "context": context_group["context"],
            "context_chunks": context_chunks,
            "token_lengths": token_lengths,
        }

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"Chunked context data saved to {OUTPUT_FILE}")
