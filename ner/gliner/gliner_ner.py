from gliner import GLiNER
from ner.types import NamedEntity
from ner.spacy.spacy_labels import spacy_labels

# Load a model
model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5", load_tokenizer=True)

def detect_entities(text: str, labels: list[str]) -> list[NamedEntity]:
    entities = model.predict_entities(text, labels, threshold=0.5)
    return [NamedEntity(name=entity['text'], label=entity['label']) for entity in entities]