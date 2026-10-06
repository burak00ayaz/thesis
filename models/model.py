from abc import ABC, abstractmethod
from enum import Enum

MAX_NEW_TOKENS = 128

class Model(Enum):
    MISTRAL = "mistral"
    PISCO = "pisco"

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
    def answer_question(
        self,
        question: str,
        context: list[str],
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        pass

def get_model(model: Model):
    if model == Model.MISTRAL:
        from models.Mistral.mistral_model import MistralModel
        return MistralModel()
    elif model == Model.PISCO:
        from models.PISCO.pisco_model import PiscoModel
        return PiscoModel()
    else:
        raise ValueError(f"Unknown model: {model}")