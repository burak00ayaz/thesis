import spacy
from ner.types import NamedEntity

nlp = spacy.load("en_core_web_sm")

def detect_entities(text: str) -> list[NamedEntity]:
    doc = nlp(text)
    return [NamedEntity(text=ent.text, label=ent.label_) for ent in doc.ents]