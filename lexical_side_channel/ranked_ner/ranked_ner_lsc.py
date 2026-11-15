from lexical_side_channel.lsc import LexicalSideChannelMethod
from tf_idf.kilt_phrase import show
from ner.gliner.gliner_ner import detect_entities as detect_entities_gliner
from evaluation.util import normalize_string

DATABASE_PATH = "tf_idf/squad_entities.sqlite"
LABELS = ["PERSON", "ORGANIZATION", "LOCATION"]

class RankedNerLSC(LexicalSideChannelMethod):
    def __init__(self):
        self.min_idf = show(db_path=DATABASE_PATH, phrase="claremont")  # Get the maximum IDF value from the database (claremont df = 200)

    def process(self, context: list[str]) -> str:
        context_text = " ".join(context)
        entities = detect_entities_gliner(context_text, labels=LABELS, threshold=0.5)

        entities = list(set([entity.name for entity in entities]))
        normalized_entities = [normalize_string(entity) for entity in entities]

        idf_scores = []
        for entity_name, normalized_name in zip(entities, normalized_entities):
            idf = show(db_path=DATABASE_PATH, phrase=normalized_name)
            idf_scores.append({"term": entity_name, "idf": idf})

        # filter out entities with higher IDF scores than the minimum IDF value
        idf_scores = [entity for entity in idf_scores if entity["idf"] > self.min_idf]

        # sort the entities by their IDF scores in descending order
        idf_scores = sorted(idf_scores, key=lambda x: x["idf"], reverse=True)
        # return top 5 entities with highest IDF scores
        top_entities = [entity["term"] for entity in idf_scores[:5]]
        return f"Entities: ({'|'.join(top_entities)})"