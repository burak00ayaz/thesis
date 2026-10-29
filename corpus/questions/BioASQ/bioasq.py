from datasets import load_dataset

from corpus.types import Triplet, Dataset

def get_triplets(split: str):
    if split not in ["factoid"]:
        raise ValueError(f"Unsupported split: {split}. Supported splits are 'factoid'. 'list' is not implemented yet.")

    ds = load_dataset("JJBauer/bioasq-14b", split=split)
    ds = ds.shuffle(seed=42)  # Shuffle the dataset for randomness

    for question in ds:
        triplet = Triplet(
            dataset=Dataset.BIOASQ_FACTOID,
            question_id=question["id"],
            context=[snippet["text"] for snippet in question["snippets"]],
            question=question["question"],
            answer=question["answer"][0][0]
        )
        yield triplet
