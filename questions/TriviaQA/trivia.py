from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sentence = "Paris is the capital of France."

def tokenize(sentence: str):
    return tokenizer.tokenize(sentence)

ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")

# print(type(ds))
# <class 'datasets.dataset_dict.DatasetDict'>

# print(ds["train"].features)
# {'question': Value('string'), 'question_id': Value('string'), 'question_source': Value('string'), 'entity_pages': {'doc_source': List(Value('string')), 'filename': List(Value('string')), 'title': List(Value('string')), 'wiki_context': List(Value('string'))}, 'search_results': {'description': List(Value('string')), 'filename': List(Value('string')), 'rank': List(Value('int32')), 'title': List(Value('string')), 'url': List(Value('string')), 'search_context': List(Value('string'))}, 'answer': {'aliases': List(Value('string')), 'normalized_aliases': List(Value('string')), 'matched_wiki_entity_name': Value('string'), 'normalized_matched_wiki_entity_name': Value('string'), 'normalized_value': Value('string'), 'type': Value('string'), 'value': Value('string')}}

# print(len(ds["train"]))
# 61888

def get_question_context_answer(example):
    question = example["question"]
    # context = example["entity_pages"]["wiki_context"] + example["search_results"]["wiki_context"]
    context = example["entity_pages"]["wiki_context"][0]
    answer = example["answer"]["value"]
    return {"question": question, "context": context, "answer": answer}


for i in range(5):
    dict_example = get_question_context_answer(ds["train"][i])
    print('Length of tokens in context: ', len(tokenize(dict_example["context"])))
    # Contexts are very long!!!
