from questions.types import Triplet, Dataset
from models.PISCO.context_chunker import get_context_chunks
import json
from pathlib import Path
from datasets import load_dataset
import hashlib
from typing import Literal

DATA_DIR = Path(__file__).parent / 'data'
DATASET_SPLIT: Literal["train", "validation"] = "train"

ENTITY_MAPPINGS_FILE = DATA_DIR / DATASET_SPLIT / 'entity_mappings.json'

with open(ENTITY_MAPPINGS_FILE, "r", encoding="utf-8") as f:
    entity_mappings = json.load(f)

def map_entities(triplet: Triplet, context_hash: str) -> Triplet:
    if context_hash in entity_mappings:
        for entity in entity_mappings[context_hash]["entities"]:
            entity_name = entity["name"]
            entity_mapping = entity["mapping"]
            triplet = Triplet(
                dataset=triplet.dataset,
                question_id=triplet.question_id,
                context=[chunk.replace(entity_name, entity_mapping) for chunk in triplet.context],
                question=triplet.question.replace(entity_name, entity_mapping),
                answer=triplet.answer.replace(entity_name, entity_mapping)
            )
        return triplet
    else:
        raise ValueError(f"No entity mapping found for context hash: {context_hash}")

def get_triplets(entity_mapping: bool = False):
    ds = load_dataset("rajpurkar/squad")["train"]

    for question in ds:
        triplet = Triplet(
            dataset=Dataset.SQUAD_MAPPED if entity_mapping else Dataset.SQUAD,
            question_id=question["id"],
            context=get_context_chunks(question["context"]),
            question=question["question"],
            answer=question["answers"]["text"][0]
        )
        if entity_mapping:
            context_hash = hashlib.sha256(question["context"].encode('utf-8')).hexdigest()
            yield map_entities(triplet, context_hash)
        else:
            yield triplet