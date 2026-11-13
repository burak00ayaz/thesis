import json
from corpus.types import Triplet, Dataset
from pathlib import Path
from models.PISCO.context_chunker import get_context_chunks

# read json file named 'natural_questions_answers.json'
with open(Path(__file__).parent / 'natural_questions_answers.json', 'r') as f:
    data = json.load(f)

def nq_original_triplets():
    for question in data:
        # return as triplets
        for annotation in question["annotations"]:
            if annotation["long_answer"] is not None and len(annotation["short_answers"]) == 1:
                context = annotation["long_answer"]
                context_chunks = get_context_chunks(context)
                if len(context_chunks) > 5:
                    print(f"Skipping question {question['id']} due to too many context chunks ({len(context_chunks)}).")
                    continue

                triplet = Triplet(
                    dataset=Dataset.NQ_GOOGLE,
                    question_id=question["id"],
                    answer=annotation["short_answers"][0],
                    question=question["question"],
                    context=context_chunks
                )

                yield triplet
                break

if __name__ == "__main__":
    nq_triplets = nq_original_triplets()

    for i, (triplet, title) in enumerate(nq_triplets):
        print(f"Triplet {i}:")
        print(f"Question ID: {triplet.question_id}")
        print(f"Context: {triplet.context}")
        print(f"Question: {triplet.question}")
        print(f"Answer: {triplet.answer}")
        print(f"Detected Title: {title}")
        print(80 * "=")
        if i == 10:
            break