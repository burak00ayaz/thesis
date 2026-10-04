from enum import Enum
from dataclasses import dataclass

class Dataset(Enum):
    SQUAD = "squad"
    SQUAD_MAPPED = "squad_mapped" #123 questions available now with entity mappings, but not all of them have been generated yet.

@dataclass
class Triplet:
    question_id: str
    context: list[str]
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


if __name__ == "__main__":
    squad_triplets = triplets(Dataset.SQUAD)
    squad_mapped_triplets = triplets(Dataset.SQUAD_MAPPED)

    for i, (triplet_real, triplet_mapped) in enumerate(zip(squad_triplets, squad_mapped_triplets)):
        print(f"Real Triplet {i}:")
        print(f"Question ID: {triplet_real.question_id}")
        print(f"Context: {triplet_real.context}")
        print(f"Question: {triplet_real.question}")
        print(f"Answer: {triplet_real.answer}")
        print(80 * "-")
        print(f"Mapped Triplet {i}:")
        print(f"Question ID: {triplet_mapped.question_id}")
        print(f"Context: {triplet_mapped.context}")
        print(f"Question: {triplet_mapped.question}")
        print(f"Answer: {triplet_mapped.answer}")
        print(80 * "=")
        if i == 122:
            break