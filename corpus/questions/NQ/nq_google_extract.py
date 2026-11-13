
from datasets import load_dataset
import json

# This tries to download the whole dataset (+50GB)
# dataset = load_dataset(
#     "google-research-datasets/natural_questions",
#     "default",
#     split="validation",
#     streaming=True
# )

dataset = load_dataset(
    "parquet",
    data_files={
        "validation": (
            "hf://datasets/google-research-datasets/"
            "natural_questions/default/"
            "validation-*.parquet"
        )
    },
    split="validation",
)

def as_records(value):
    """Handle Hugging Face list-of-dicts or dict-of-lists."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if not value:
            return []
        keys = list(value)
        return [
            {key: value[key][i] for key in keys}
            for i in range(len(value[keys[0]]))
        ]
    return []

def extract_span(tokens, span):
    """Convert original NQ token offsets to readable text."""
    if not span:
        return None

    start = span.get("start_token", -1)
    end = span.get("end_token", -1)

    if start < 0 or end <= start:
        return None

    selected = tokens[start:end]

    return " ".join(
        token["token"]
        for token in selected
        if not token["is_html"]
    ).strip() or None

def extract_qa(example):
    tokens = as_records(example["document"]["tokens"])
    annotations = as_records(example["annotations"])

    results = []

    for annotation in annotations:
        long_answer = extract_span(
            tokens,
            annotation["long_answer"]
        )

        short_answers = [
            text
            for span in as_records(annotation["short_answers"])
            if (text := extract_span(tokens, span))
        ]

        results.append({
            "long_answer": long_answer,
            "short_answers": short_answers,
            "yes_no_answer": annotation["yes_no_answer"]
        })

    #extract wikipedia document title from the document
    document = example["document"]["html"]
    #find the title in the document
    start_index = document.find("<TITLE>") + len("<TITLE>")
    end_index = document.find("</TITLE>")
    title = document[start_index:end_index]
    title = title.removesuffix(" - Wikipedia")

    return {
        "id": example["id"],
        "question": example["question"]["text"],
        "annotations": results,
        "wikipedia_document_title": title
    }

def is_unanswerable(record):
    return all(
        ann["long_answer"] is None
        or len(ann["short_answers"]) == 0
        for ann in record["annotations"]
    )

results = []

for i, example in enumerate(dataset):
    results.append(extract_qa(example))
    print(f"Processed {i + 1} examples", end="\r")

answerable_questions = [r for r in results if not is_unanswerable(r)]

# Save to JSON
with open("natural_questions_answers.json", "w", encoding="utf-8") as f:
    json.dump(answerable_questions, f, ensure_ascii=False, indent=2)
