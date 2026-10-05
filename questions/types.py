from enum import Enum
from dataclasses import dataclass

class Dataset(Enum):
    SQUAD = "squad"
    SQUAD_MAPPED = "squad_mapped" #123 questions available now with entity mappings, but not all of them have been generated yet.
    SYNTHWORLDS_RM = "synthworlds_rm"
    SYNTHWORLDS_SM = "synthworlds_sm"

@dataclass
class Triplet:
    dataset: Dataset
    question_id: str
    context: list[str]
    question: str
    answer: str

DATASET_PAIRS = {
    Dataset.SQUAD: Dataset.SQUAD_MAPPED,
    Dataset.SQUAD_MAPPED: Dataset.SQUAD,
    Dataset.SYNTHWORLDS_RM: Dataset.SYNTHWORLDS_SM,
    Dataset.SYNTHWORLDS_SM: Dataset.SYNTHWORLDS_RM,
}