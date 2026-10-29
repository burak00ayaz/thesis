from pathlib import Path
import json

from corpus.types import Dataset, Triplet
from corpus.questions.questions import triplets
from ner.gliner.gliner_ner import detect_entities as detect_entities_gliner
# from ner.spacy.spacy_labels import spacy_labels

##################################################################################################
# NER Configuration
##################################################################################################

GLINER_THRESHOLD = 0.5

# We can implement a label selection mechanism if needed
# For now, use the same labels as Apple study 5 entity types
LABELS = ["PERSON", "TIME", "NUMERIC", "ORGANIZATION", "LOCATION"]
BIOASQ_LABELS = ["GENE", "DISEASE", "CHEMICAL", "ORGANISM", "CELL_TYPE", "CELL_LINE", "TISSUE", "MEDICAL"]

##################################################################################################
# Script Configuration
##################################################################################################

STOP_AFTER = 190
DATASET = Dataset.BIOASQ_FACTOID
OUTPUT_FILE = Path(__file__).parent / f"entity_questions/{DATASET.name.lower()}.json"
APPEND_MODE = True  # If True, append to existing file; if False, overwrite

###################################################################################################

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

if APPEND_MODE and OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        entity_questions = json.load(f)
        latest_question_id = list(entity_questions.keys())[-1]  # Get the last question ID
        skip_until_id = latest_question_id
        print(f"Resuming from last processed question ID: {skip_until_id}")
else:
    entity_questions = {}
    skip_until_id = None

count = 0

for i, (triplet, _) in enumerate(triplets(DATASET, skip_until_id=skip_until_id)):
    print(f"Processing triplet {i}...")
    context = " ".join(triplet.context)
    gliner_entities = detect_entities_gliner(context, labels=BIOASQ_LABELS, threshold=GLINER_THRESHOLD)
    
    if triplet.answer in [entity.name for entity in gliner_entities]:
        detected_entity_gliner = next(entity for entity in gliner_entities if entity.name == triplet.answer)
        print(f"GLiNER detected the answer entity: {detected_entity_gliner}")
        print(f"Context: {context}")
        print(f"Answer: {triplet.answer}")
        print(80 * "-")
        count += 1
        entity_questions[triplet.question_id] = {
            "count": entity_questions.keys().__len__() + 1,
            "context": triplet.context,
            "question": triplet.question,
            "answer_entity": triplet.answer,
            "label": detected_entity_gliner.label
        }

    if count >= STOP_AFTER:
        break


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(entity_questions, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(entity_questions.keys())} triplets with detected answer entities to {OUTPUT_FILE}")


print(f"Total triplets processed: {i + 1}")
print(f"Total triplets with detected answer entities: {count}")

# group detected entities by label
grouped_entities = {}
for question_id, entity_question in entity_questions.items():
    if entity_question["label"] not in grouped_entities:
        grouped_entities[entity_question["label"]] = []
    grouped_entities[entity_question["label"]].append(entity_question["answer_entity"])

print("Grouped entities by label:")
for label, names in grouped_entities.items():
    print(f"  {label}: {', '.join(names)}")
    print(f"  Count: {len(names)}")
    print(80 * "-")
