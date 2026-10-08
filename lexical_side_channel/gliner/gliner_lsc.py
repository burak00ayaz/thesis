from lexical_side_channel.lsc import LexicalSideChannelMethod
from ner.gliner.gliner_ner import detect_entities


class GlinerLSC(LexicalSideChannelMethod):
    def process(self, context: list[str]) -> str:
        context_text = " ".join(context)
        entities = detect_entities(context_text)
        entity_names = [entity.name for entity in entities]
        return f"Entities: ({'|'.join(entity_names)})"