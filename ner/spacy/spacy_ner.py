import spacy
from ner.types import NamedEntity

nlp = spacy.load("en_core_web_sm")

def detect_entities(text: str) -> list[NamedEntity]:
    doc = nlp(text)
    return [NamedEntity(name=ent.text, label=ent.label_, start_char=ent.start_char, end_char=ent.end_char) for ent in doc.ents]