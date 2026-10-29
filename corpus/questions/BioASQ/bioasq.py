from datasets import load_dataset

from corpus.types import Triplet, Dataset
from models.PISCO.context_chunker import get_context_chunks

def get_triplets(split: str):
    if split not in ["factoid"]:
        raise ValueError(f"Unsupported split: {split}. Supported splits are 'factoid'. 'list' is not implemented yet.")

    ds = load_dataset("JJBauer/bioasq-14b", split=split)
    ds = ds.shuffle(seed=42)  # Shuffle the dataset for randomness

    for question in ds:
        snippets = [snippet["text"] for snippet in question["snippets"]]
        context = " ".join(snippets)
        context_chunks = get_context_chunks(context)

        if len(context_chunks) > 5:
            print(f"Skipping question {question['id']} due to too many context chunks ({len(context_chunks)}).")
            continue

        triplet = Triplet(
            dataset=Dataset.BIOASQ_FACTOID,
            question_id=question["id"],
            context=context_chunks,
            question=question["question"],
            answer=question["answer"][0][0]
        )
        yield triplet
