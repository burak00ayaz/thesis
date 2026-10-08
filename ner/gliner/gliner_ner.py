from gliner import GLiNER
from ner.types import NamedEntity

# Load a model
model = GLiNER.from_pretrained("gliner-community/gliner_xxl-v2.5", load_tokenizer=True)

labels_1 = ["person", "organization", "date", "location", "event"]
labels_2 = ["person", "organization", "location", "date", "time", "faculty", "number", "quantity", "event", "product", "work of art", "money", "percentage"]

# Recommended by ChatGPT
# PERSON
# ORGANIZATION
# LOCATION
# DATE
# TIME
# NUMBER
# QUANTITY
# EVENT
# PRODUCT
# WORK
# NATIONALITY_GROUP
# OTHER_NAMED_ENTITY

def detect_entities(text: str) -> list[NamedEntity]:
    entities = model.predict_entities(text, labels_2, threshold=0.1)
    return [NamedEntity(text=entity['text'], label=entity['label']) for entity in entities]