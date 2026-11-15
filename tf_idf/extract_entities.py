from corpus.types import Dataset
from corpus.questions.questions import triplets
from ner.gliner.gliner_ner import detect_entities as detect_entities_gliner
from evaluation.util import normalize_string


squad_ds = triplets(Dataset.SQUAD, entity_questions=False)
LABELS = ["PERSON", "ORGANIZATION", "LOCATION"]

detected_entities = set()


for i, (triplet, _) in enumerate(squad_ds):
    print(f"Processing triplet {i}...", end="\r")
    context = " ".join(triplet.context)
    gliner_entities = detect_entities_gliner(context, labels=LABELS, threshold=0.5)
    normalized_entities = [normalize_string(entity.name) for entity in gliner_entities]
    detected_entities.update(normalized_entities)

    if i >= 1000:
        break

with open("tf_idf/squad_entities.txt", "w", encoding="utf-8") as f:
    for entity in sorted(detected_entities):
        f.write(f"{entity}\n")