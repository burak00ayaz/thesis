import spacy
from lexical_side_channel.lsc import LexicalSideChannelMethod

nlp = spacy.load("en_core_web_sm")

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

class SpacyNER(LexicalSideChannelMethod):
    def process(self, context: list[str]) -> str:
        context_text = " ".join(context)
        doc = nlp(context_text)
        entities = [ent.text for ent in doc.ents]
        return f"Entities: ({'|'.join(entities)})"