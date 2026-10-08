from enum import Enum
from abc import ABC, abstractmethod

# List all the LSC baseline methods here.
class LexicalSideChannelBaseline(Enum):
    SPACY_LSC = "spacy_lsc"
    GLINER_LSC = "gliner_lsc"

class LexicalSideChannelMethod(ABC):
    @abstractmethod
    def process(self, context: list[str]) -> str:
        pass

def get_lexical_side_channel_method(baseline: LexicalSideChannelBaseline) -> LexicalSideChannelMethod:
    if baseline == LexicalSideChannelBaseline.SPACY_LSC:
        from lexical_side_channel.spacy.spacy_lsc import SpacyLSC
        return SpacyLSC()
    elif baseline == LexicalSideChannelBaseline.GLINER_LSC:
        from lexical_side_channel.gliner.gliner_lsc import GlinerLSC
        return GlinerLSC()
    else:
        raise ValueError(f"Unsupported lexical side channel baseline: {baseline}")





