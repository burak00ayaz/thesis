from lexical_side_channel.lsc import LexicalSideChannelMethod
from ner.spacy.spacy_ner import detect_entities

all_labels = """
CARDINAL: Numerals that do not fall under another type
DATE: Absolute or relative dates or periods
EVENT: Named hurricanes, battles, wars, sports events, etc.
FAC: Buildings, airports, highways, bridges, etc.
GPE: Countries, cities, states
LANGUAGE: Any named language
LAW: Named documents made into laws.
LOC: Non-GPE locations, mountain ranges, bodies of water
MONEY: Monetary values, including unit
NORP: Nationalities or religious or political groups
ORDINAL: "first", "second", etc.
ORG: Companies, agencies, institutions, etc.
PERCENT: Percentage, including "%"
PERSON: People, including fictional
PRODUCT: Objects, vehicles, foods, etc. (not services)
QUANTITY: Measurements, as of weight or distance
TIME: Times smaller than a day
WORK_OF_ART: Titles of books, songs, etc.
"""

class SpacyLSC(LexicalSideChannelMethod):
    def process(self, context: list[str]) -> str:
        context_text = " ".join(context)
        entities = detect_entities(context_text)
        entity_names = [entity.name for entity in entities]
        return f"Entities: ({'|'.join(entity_names)})"