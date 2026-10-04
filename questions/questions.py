from enum import Enum
from dataclasses import dataclass

class Dataset(Enum):
    SQUAD = "squad"
    SQUAD_MAPPED = "squad_mapped"

@dataclass
class Triplet:
    context: str
    question: str
    answer: str

def triplets(dataset: Dataset):
    if dataset == Dataset.SQUAD:
        from questions.SQUAD.squad import get_triplets
        return get_triplets(entity_mapping=False)
    elif dataset == Dataset.SQUAD_MAPPED:
        from questions.SQUAD.squad import get_triplets
        return get_triplets(entity_mapping=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")