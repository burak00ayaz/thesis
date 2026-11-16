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
    from models.PISCO.context_chunker import get_context_chunks

    for i, question in enumerate(ds):
        if is_single_answer(question['id']) or True:  # Include all questions for validation
            context_chunks = get_context_chunks(question['context'])
            yield Triplet(
                dataset=Dataset.SQUAD_VALIDATION,
                question_id=question['id'],
                context=context_chunks,
                question=question['question'],
                answer=question['answers']['text'][0]  # Assuming single answer
            )


if __name__ == "__main__":
    from models.PISCO.context_chunker import get_context_chunks
    from transformers import AutoTokenizer

    # tokenizer of the PISCO backbone model
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    def token_len(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    total_token_count = 0
    total_question_count = 0
    token_lengths = []

    for i, triplet in enumerate(get_triplets()):
        context = " ".join(triplet.context)
        token_length = token_len(context)
        print(f"Token length: {token_length}")
        total_token_count += token_length
        total_question_count += 1
        token_lengths.append(token_length)

    average_token_length = total_token_count / total_question_count if total_question_count > 0 else 0
    print(f"Average token length of contexts: {average_token_length}")

    median_token_length = sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0
    print(f"Median token length of contexts: {median_token_length}")

    # plot the distribution of token lengths and save plot as a PNG file
    import matplotlib.pyplot as plt

    plt.hist(token_lengths, bins=30, edgecolor='black')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Context Token Lengths in SQuAD Validation')

    plt.savefig("squad_validation_token_lengths.png")
