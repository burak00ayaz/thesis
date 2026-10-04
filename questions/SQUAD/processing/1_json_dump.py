from datasets import load_dataset
import hashlib
from pathlib import Path
import json
from typing import Literal

# Group the questions by their corresponding context and store them in a dictionary, 
# where each value contains the context, questions, and answers.

DATASET_NAME = "rajpurkar/squad"
DATASET_SPLIT: Literal["train", "validation"] = "train"

DATA_DIR = Path(__file__).parent.parent / 'data'
OUTPUT_FILE = DATA_DIR / DATASET_SPLIT / 'context_groups.json'

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("rajpurkar/squad")[DATASET_SPLIT]

def hash_string(input: str) -> str:
    return hashlib.sha256(input.encode('utf-8')).hexdigest()

def process_squad_dataset():
    dataset = {}
    for i, example in enumerate(ds):
        print(f"Processing example {i}/{len(ds)}")
        question, context, answer = example["question"], example["context"], example["answers"]["text"][0] 
        context_hash = hash_string(context)
        if context_hash not in dataset:
            dataset[context_hash] = {
                "context": context,
                "questions": [question],
                "answers": [answer],
            }
        else:
            dataset[context_hash]["answers"].append(answer)
            dataset[context_hash]["questions"].append(question)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    process_squad_dataset()
    print(f"Dataset dumped to {OUTPUT_FILE}")