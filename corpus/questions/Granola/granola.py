import json
import os
from pathlib import Path

from datasets import load_dataset
from corpus.types import Triplet, Dataset
from models.PISCO.context_chunker import get_context_chunks

CONTEXT_TEMPLATES_FILE = "relation_context_templates.json"
WIKIPEDIA_ARTICLES_FILE = "articles/wikipedia_articles.json"


MODE = "wikipedia"  # or "synthetic" for synthetic context generation


def get_triplets():
    ds = load_dataset("google/granola-entity-questions", token=True)["train"]
    ds = ds.shuffle(seed=42)  # Shuffle the dataset for randomness

    if MODE == "synthetic":
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            relation_context_templates_path = os.path.join(current_dir, CONTEXT_TEMPLATES_FILE)
            with open(relation_context_templates_path, "r") as f:
                relation_context_templates = json.load(f)
        except:
            raise FileNotFoundError("Context templates file could not be loaded.")

    if MODE == "wikipedia":
        try:
            wikipedia_articles_path = Path(__file__).parent / WIKIPEDIA_ARTICLES_FILE
            with open(wikipedia_articles_path, "r") as f:
                wikipedia_articles = json.load(f)
        except Exception as e:
            print(f"ERROR: {e}")
            raise FileNotFoundError("Wikipedia articles file could not be loaded.")

    for i, example in enumerate(ds):
        question_id = str(i + 1)

        if MODE == "synthetic":
            context_template = relation_context_templates.get(example["relation"], None).get("synthetic_context", None)
            if context_template is None:
                raise ValueError(f"Context template for relation '{example['relation']}' not found in {CONTEXT_TEMPLATES_FILE}.")
                exit(1)
            context = context_template.format(
                QE=example["question_entity"],
                AE=example["answer"]
            )
            context_chunks = [context]
        elif MODE == "wikipedia":
            article_path = wikipedia_articles.get(example["question_entity_qid"], {}).get("article_path", "")
            if not article_path or not os.path.exists(article_path):
                print(f"WARN: Article for {example['question_entity']} {example['question_entity_qid']} not found. Skipping.")
                continue
            else: 
                with open(article_path, "r") as f:
                    article = f.read()
                context_chunks = get_context_chunks(article)
                if len(context_chunks) > 10:
                    print(f"WARN: Article for {example['question_entity']} {example['question_entity_qid']} has more than 10 context chunks ({len(context_chunks)}). Skipping.")
                    continue
                
        yield Triplet(dataset=Dataset.GRANOLA, question_id=question_id, context=context_chunks, question=example["question"], answer=example["answer"])