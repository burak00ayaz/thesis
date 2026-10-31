from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import time
import torch


def wikipedia_title_to_wikidata_id(title: str) -> str | None:
    response = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "sites": "enwiki",
            "titles": title,
            "props": "info",
            "format": "json",
        },
        headers={
            "User-Agent": "ThesisEntityLinking/0.1 (contact: your-email@example.com)"
        },
        timeout=15,
    )
    response.raise_for_status()

    entities = response.json().get("entities", {})

    for entity_id, entity in entities.items():
        if "missing" not in entity:
            return entity_id

    return None



# OPTIONAL: load the prefix tree (trie), you need to additionally download
# https://huggingface.co/facebook/genre-linking-blink/blob/main/trie.py and 
# https://huggingface.co/facebook/genre-linking-blink/blob/main/kilt_titles_trie_dict.pkl
# import pickle
# from trie import Trie
# with open("kilt_titles_trie_dict.pkl", "rb") as f:
#     trie = Trie.load_from_dict(pickle.load(f))

tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink").eval()

sentences = ["Einstein was a [START_ENT] German [END_ENT] physicist."]

outputs = model.generate(
    **tokenizer(sentences, return_tensors="pt"),
    num_beams=5,
    num_return_sequences=5,
    max_new_tokens=32,
    # OPTIONAL: use constrained beam search
    # prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
    return_dict_in_generate=True,
    output_scores=True,
)

predictions = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

scores = outputs.sequences_scores

relative_probs = torch.softmax(scores, dim=0)

top_entity_id = wikipedia_title_to_wikidata_id(predictions[0])
print(f"Top prediction: {predictions[0]} | Top entity ID: {top_entity_id}")

for prediction, score, prob in zip(
    predictions,
    scores,
    relative_probs,
):
    print(
        f"{score.item():.4f} | "
        f"{prob.item():.2%} | "
        f"{prediction}"
    )