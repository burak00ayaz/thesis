from datasets import load_dataset
from questions.types import Triplet, Dataset
from models.PISCO.context_chunker import get_context_chunks

def get_triplets(dataset: Dataset):
    if dataset == Dataset.SYNTHWORLDS_SM:
        ds = load_dataset("kenqgu/SynthWorlds", "qa-sm", split="test")
    elif dataset == Dataset.SYNTHWORLDS_RM:
        ds = load_dataset("kenqgu/SynthWorlds", "qa-rm", split="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    for example in ds:
        yield Triplet(
            dataset=dataset,
            question_id=example["instance_id"].rsplit("-", 1)[0],
            context=get_context_chunks(example["gold_docs"][0]),
            question=example["query"],
            answer=example["gold_answers"][0]
        )