import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()


def masked_word_probability(text, target_word):
    """
    Example:
        text = "Barack Obama was born in [MASK]."
        target_word = "hawaii"
    """

    inputs = tokenizer(text, return_tensors="pt")

    # Find [MASK]
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]

    if len(mask_positions) != 1:
        raise ValueError("Expected exactly one [MASK] token.")

    mask_position = mask_positions.item()

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Vocabulary logits at [MASK]
    logits = outputs.logits[0, mask_position]

    # Convert logits -> probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Token ID of target
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)

    if len(target_ids) != 1:
        raise ValueError(
            f"'{target_word}' is represented by {len(target_ids)} tokens: "
            f"{tokenizer.convert_ids_to_tokens(target_ids)}"
        )

    target_id = target_ids[0]

    return probabilities[target_id].item()

def top_mask_predictions(text, k=20):
    inputs = tokenizer(text, return_tensors="pt")

    mask_position = (
        inputs["input_ids"] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[1].item()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, mask_position]
    probs = torch.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, k)

    for prob, token_id in zip(top_probs, top_ids):
        token = tokenizer.decode([token_id])
        print(f"{token:20s} {prob.item():.6f}")

#top_mask_predictions(
#    "Barack Obama was born in [MASK].",
#    k=20
#)

for text in [
    "Barack Obama was born in [MASK].",
    "Barack Obama was born in Honolulu, [MASK].",
    "Barack Obama was born in the state of [MASK].",
    "The birthplace of Barack Obama is [MASK].",
]:
    print("\n" + text)
    top_mask_predictions(text, 10)

exit(0)

p = masked_word_probability(
    "Barack Obama was born in [MASK].",
    "hawaii",
)

print(p)

p = masked_word_probability(
    "Barack Obama was born in [MASK].",
    "istanbul",
)

print(p)
