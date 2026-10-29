import json
from pathlib import Path

from corpus.types import Dataset, Triplet

def _load_entity_questions(dataset: Dataset) -> dict:
    """
    Load the question IDs of triplets with detected answer entities from a JSON file.
    """
    output_file = Path(__file__).parent / f"entity_questions/{dataset.name.lower()}.json"
    if not output_file.exists():
        raise FileNotFoundError(f"Entity questions file not found: {output_file}")

    with open(output_file, "r", encoding="utf-8") as f:
        entity_questions = json.load(f)

    return entity_questions

def filtering_wrapper(triplets, dataset: Dataset, filter_entity_questions=False):
    if not filter_entity_questions:
        for triplet in triplets:
            yield (triplet, None)
    else:
        entity_questions = _load_entity_questions(dataset)
        entity_question_ids = set(entity_questions.keys())
        for triplet in triplets:
            if triplet.question_id in entity_question_ids:
                yield (triplet, entity_questions[triplet.question_id]["label"])