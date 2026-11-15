from enum import Enum
from abc import ABC, abstractmethod

# List all the LSC baseline methods here.
class LexicalSideChannelBaseline(Enum):
    SPACY_LSC = "spacy_lsc"
    GLINER_LSC = "gliner_lsc"
    NUMERICAL_LSC = "numerical_lsc"
    TF_IDF_LSC = "tf_idf_lsc"
    RANKED_NER_LSC = "ranked_ner_lsc"

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
    elif baseline == LexicalSideChannelBaseline.NUMERICAL_LSC:
        from lexical_side_channel.numerical.numerical_detector import NumericalSideChannel
        return NumericalSideChannel()
    elif baseline == LexicalSideChannelBaseline.TF_IDF_LSC:
        from lexical_side_channel.tf_idf.kilt_lsc import KiltLSC
        return KiltLSC()
    elif baseline == LexicalSideChannelBaseline.RANKED_NER_LSC:
        from lexical_side_channel.ranked_ner.ranked_ner_lsc import RankedNerLSC
        return RankedNerLSC()
    else:
        raise ValueError(f"Unsupported lexical side channel baseline: {baseline}")





