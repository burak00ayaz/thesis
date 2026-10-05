from abc import ABC, abstractmethod

MAX_NEW_TOKENS = 128

class Model(ABC):
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