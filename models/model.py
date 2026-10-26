from abc import ABC, abstractmethod
from enum import Enum
from corpus.questions.types import Triplet

class Model(Enum):
    MISTRAL = "mistral"
    PISCO = "pisco"
    XRAG = "xrag"

class ModelAbstract(ABC):
    def __init__(
        self,
        backbone_model: str,
        soft_compression: str | None = None,
        compression_ratio: float | None = None,
    ):
        self.backbone_model = backbone_model
        self.soft_compression = soft_compression
        self.compression_ratio = compression_ratio

    @abstractmethod
    def answer_question(self, triplet: Triplet) -> str:
        pass

    @abstractmethod
    def answer_questions_batch(self, triplets: list[Triplet]) -> list[str]:
        pass

def get_model(model: Model):
    if model == Model.MISTRAL:
        from models.Mistral.mistral_model import MistralModel
        return MistralModel()
    elif model == Model.PISCO:
        from models.PISCO.pisco_model import PiscoModel
        return PiscoModel()
    elif model == Model.XRAG:
        from models.xRAG.xrag_model import xRAGModel
        return xRAGModel()
    else:
        raise ValueError(f"Unknown model: {model}")