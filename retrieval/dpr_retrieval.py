from datasets import load_dataset

wiki = load_dataset(
    "facebook/wiki_dpr",
    "psgs_w100.no_embeddings",
    split="train"
)