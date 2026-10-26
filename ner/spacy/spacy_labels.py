# python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('\n'.join(nlp.get_pipe('ner').labels))"
# python -c "import spacy; nlp=spacy.load('en_core_web_sm'); [print(f'{label:15} {spacy.explain(label)}') for label in nlp.get_pipe('ner').labels]"

# CARDINAL        Numerals that do not fall under another type
# DATE            Absolute or relative dates or periods
# EVENT           Named hurricanes, battles, wars, sports events, etc.
# FAC             Buildings, airports, highways, bridges, etc.
# GPE             Countries, cities, states
# LANGUAGE        Any named language
# LAW             Named documents made into laws.
# LOC             Non-GPE locations, mountain ranges, bodies of water
# MONEY           Monetary values, including unit
# NORP            Nationalities or religious or political groups
# ORDINAL         "first", "second", etc.
# ORG             Companies, agencies, institutions, etc.
# PERCENT         Percentage, including "%"
# PERSON          People, including fictional
# PRODUCT         Objects, vehicles, foods, etc. (not services)
# QUANTITY        Measurements, as of weight or distance
# TIME            Times smaller than a day
# WORK_OF_ART     Titles of books, songs, etc.

spacy_labels = [
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]