from datasets import load_dataset
from transformers import AutoTokenizer
import hashlib
from pathlib import Path
import json

CURRENT_DIR = Path(__file__).parent

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sentence = "Paris is the capital of France."

def tokenize(sentence: str):
    return tokenizer.tokenize(sentence)


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("rajpurkar/squad")

# print(type(ds))
# <class 'datasets.dataset_dict.DatasetDict'>

# print(ds["train"].features)
# {'id': Value('string'), 'title': Value('string'), 'context': Value('string'), 'question': Value('string'), 'answers': {'text': List(Value('string')), 'answer_start': List(Value('int32'))}}

# print(len(ds["train"]))
# 87599

# print(f"Number of examples with context length <= 128 tokens: {count}")
# 19705

# implement a generator that yields question, context, and answer for each example in the dataset
def get_question_context_answer_triples():
    for i in range(len(ds["train"])):
        example = ds["train"][i]
        context = example["context"]
        length = len(tokenize(context))
        if length > 128:
            continue
        question = example["question"]
        answer = example["answers"]["text"][0]  # Get the first answer from the list
        yield {"question": question, "context": context, "answer": answer}

def hash_context(context: str) -> str:
    return hashlib.sha256(context.encode()).hexdigest()


# read 'squad_entity_map.json' and check if the context hash exists
try:
    with open(CURRENT_DIR / 'squad_entity_mappings_filled.json', 'r') as f:
        entity_map = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("squad_entity_mappings_filled.json not found. Please create the file with the appropriate mappings.")


def map_entities_in_triple(question: str, context: str, answer: str):
    context_hash = hash_context(context)

    if context_hash in entity_map:
        entity = entity_map[context_hash]
        mappings = entity_map[context_hash]["mappings"]
        mapping_keys = list(mappings.keys())
        for mapping_key in mapping_keys:
            question = question.replace(mapping_key, mappings[mapping_key])
            answer = answer.replace(mapping_key, mappings[mapping_key])
            context = context.replace(mapping_key, mappings[mapping_key])

        return {
            "question": question,
            "context": context,
            "answer": answer,
        }
    else:
        raise ValueError(f"Context hash {context_hash} not found in entity map.")

# process the dataset and add contexts to squad_entity_mappings.json if they are not already present
def process_squad_dataset():
    try:
        with open(CURRENT_DIR / 'squad_entity_mappings.json', 'r') as f:
            entity_map = json.load(f)
    except FileNotFoundError:
        entity_map = {}

    for i, example in enumerate(get_question_context_answer_triples()):
        print(f"Processing example {i}...")
        question, context, answer = example["question"], example["context"], example["answer"]
        context_hash = hash_context(context)
        if context_hash not in entity_map:
            # Add the context and a placeholder mapping to the entity map
            entity_map[context_hash] = {
                "context": context,
                "questions": [question],
                "answers": [answer],
                "mappings": {
                    "PLACEHOLDER_ENTITY": "REPLACEMENT_ENTITY"
                }
            }
        else:
            entity_map[context_hash]["answers"].append(answer)
            entity_map[context_hash]["questions"].append(question)

    # Write the updated entity map back to the file
    with open(CURRENT_DIR / 'squad_entity_mappings.json', 'w') as f:
        json.dump(entity_map, f)

if __name__ == "__main__":
    process_squad_dataset()