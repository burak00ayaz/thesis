from questions.questions import Triplet
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'

# Temporary
FILE = DATA_DIR / 'squad_entity_mappings.json'


def map_entities(triplet: Triplet, entity_map: dict) -> Triplet:
    mapping_keys = list(entity_map.keys())
    for mapping_key in mapping_keys:
        triplet = Triplet(
            triplet.context.replace(mapping_key, entity_map[mapping_key]),
            triplet.question.replace(mapping_key, entity_map[mapping_key]),
            triplet.answer.replace(mapping_key, entity_map[mapping_key])
        )
    return triplet

def get_triplets(entity_mapping: bool = False):
    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for chunk in data:
        context = chunk["context"]
        for question, answer in zip(chunk["questions"], chunk["answers"]):
            if entity_mapping:
                yield map_entities(Triplet(context, question, answer), chunk["mappings"])
            else:
                yield Triplet(context, question, answer)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")