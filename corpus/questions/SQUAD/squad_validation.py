from datasets import load_dataset
from corpus.types import Triplet, Dataset
from evaluation.util import normalize_string


DATASET_NAME = "rajpurkar/squad"
DATASET_SPLIT = "validation"


print(f"Loading dataset {DATASET_NAME} with split {DATASET_SPLIT}...")
ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)  # Shuffle the dataset for randomness


print(f"Building question ID to index mapping...")
id_to_idx = {
    question_id: idx
    for idx, question_id in enumerate(ds["id"])
}
print(f"Mapping built with {len(id_to_idx)} entries.")


def get_by_question_id(question_id: str) -> Triplet:
    if question_id not in id_to_idx:
        raise ValueError(f"Question ID {question_id} not found.")
    return ds[id_to_idx[question_id]]


def is_single_answer(question_id: str) -> bool:
    question = get_by_question_id(question_id)
    answers = question["answers"]["text"]
    answers_set = set(normalize_string(ans) for ans in answers)
    return len(answers_set) == 1


# load the SQuAD dataset using the Hugging Face datasets library
def get_triplets():
    for i, question in enumerate(ds):
        if is_single_answer(question['id']):
            yield Triplet(
                dataset=Dataset.SQUAD_VALIDATION,
                question_id=question['id'],
                context=[question['context']],
                question=question['question'],
                answer=question['answers']['text'][0]  # Assuming single answer
            )


if __name__ == "__main__":
    get_triplets()


