from ner.spacy.spacy_ner import detect_entities as spacy_detect_entities
from ner.gliner.gliner_ner import detect_entities as gliner_detect_entities
from questions.types import Dataset, Triplet
from questions.questions import triplets
import textwrap

squad_triplets = triplets(Dataset.SQUAD)
examples = 0
latest_context_text = ""

for i, triplet in enumerate(squad_triplets):
    context_text = " ".join(triplet.context)
    if context_text == latest_context_text:
        continue

    latest_context_text = context_text
    spacy_entities = spacy_detect_entities(context_text)
    gliner_entities = gliner_detect_entities(context_text)

    # PERSON test
    spacy_entities = [entity for entity in spacy_entities if entity.label == "PERSON"]
    gliner_entities = [entity for entity in gliner_entities if entity.label == "person"]

    print(f"Triplet {i}:")
    print(f"Context: {textwrap.fill(context_text, width=100)}")
    print(80 * "-")
    print("Spacy Entities:")
    for entity in spacy_entities:
        print(f"Text: {entity.text}, Label: {entity.label}")
    print(80 * "-")
    print("GLiNER Entities:")
    for entity in gliner_entities:
        print(f"Text: {entity.text}, Label: {entity.label}")
    print(80 * "=")

    examples += 1
    if examples == 10:
        break

