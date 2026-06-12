from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sentence = "Paris is the capital of France."

def tokenize(sentence: str):
    return tokenizer.tokenize(sentence)


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("rajpurkar/squad")

# print(type(ds))
# <class 'datasets.dataset_dict.DatasetDict'>

# print(ds["train"].features)
# {'id': Value('string'), 'title': Value('string'), 'context': Value('string'), 'question': Value('string'), 'answers': {'text': List(Value('string')), 'answer_start': List(Value('int32'))}}

# print(len(ds["train"]))
# 87599

# print(f"Number of examples with context length <= 128 tokens: {count}")
# 19705

# implement a generator that yields question, context, and answer for each example in the dataset
def get_question_context_answer_triples():
    for i in range(len(ds["train"])):
        example = ds["train"][i]
        context = example["context"]
        length = len(tokenize(context))
        if length > 128:
            continue
        question = example["question"]
        answer = example["answers"]["text"][0]  # Get the first answer from the list
        yield {"question": question, "context": context, "answer": answer}



for i, triple in enumerate(get_question_context_answer_triples()):
    print(f"Example {i}:")
    print("Question:", triple["question"])
    print("Context:", triple["context"])
    print("Answer:", triple["answer"])
    print()