from enum import Enum
from abc import ABC, abstractmethod

# List all the LSC baseline methods here.
class LexicalSideChannelBaseline(Enum):
    SPACY_NER = "spacy_ner"

class LexicalSideChannelMethod(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        pass

def get_lexical_side_channel_method(baseline: LexicalSideChannelBaseline) -> LexicalSideChannelMethod:
    if baseline == LexicalSideChannelBaseline.SPACY_NER:
        from lexical_side_channel.spacy.spacy_ner import SpacyNER
        return SpacyNER()
    else:
        raise ValueError(f"Unsupported lexical side channel baseline: {baseline}")





