import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
)

model.eval()


def word_surprisal(context: str, word: str):
    # Important: include the space before the word if this is how
    # the word occurs naturally in the sentence.
    full_text = context + word

    context_ids = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids

    # Everything appearing after the context tokenization
    # is treated as belonging to the target word.
    target_ids = full_ids[0, context_ids.shape[1]:]

    with torch.no_grad():
        outputs = model(full_ids)

    logits = outputs.logits

    surprisals = []

    start = context_ids.shape[1]

    for token_position in range(start, full_ids.shape[1]):
        # logits at position i-1 predict token at position i
        prediction_logits = logits[0, token_position - 1]

        log_probs = torch.log_softmax(prediction_logits, dim=-1)

        actual_token_id = full_ids[0, token_position]

        log_prob = log_probs[actual_token_id].item()

        # nats -> bits
        surprisal_bits = -log_prob / math.log(2)

        surprisals.append(surprisal_bits)

    return {
        "tokens": tokenizer.convert_ids_to_tokens(target_ids),
        "token_surprisals": surprisals,
        "total_surprisal": sum(surprisals),
    }

result1 = word_surprisal(
    "The capital of France is",
    " Paris"
)

result2 = word_surprisal(
    "The capital of France is",
    " Qwen2.5"
)

print(result1)
print(result2)