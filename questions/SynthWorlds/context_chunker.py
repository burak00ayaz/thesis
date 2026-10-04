from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import re

CHUNK_SIZE = 128

# tokenizer of the PISCO backbone model
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2"
)

nltk.download("punkt")


def token_len(text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def split_oversized_segment(text: str, max_tokens: int) -> list[str]:
    """
    Recursively split a long segment using increasingly weaker boundaries.
    Hard token splitting is only the final fallback.
    """

    if token_len(text) <= max_tokens:
        return [text]

    # Prefer stronger semantic boundaries first
    for separator_pattern in [
        r"(?<=;)\s+",   # semicolon
        r"(?<=,)\s+",   # comma
    ]:
        parts = re.split(separator_pattern, text)

        if len(parts) > 1:
            chunks = []
            current = []

            for part in parts:
                candidate = " ".join(current + [part])

                if token_len(candidate) <= max_tokens:
                    current.append(part)
                else:
                    if current:
                        chunks.append(" ".join(current))

                    # This piece alone is still too large
                    if token_len(part) > max_tokens:
                        chunks.extend(
                            split_oversized_segment(
                                part,
                                max_tokens,
                            )
                        )
                        current = []
                    else:
                        current = [part]

            if current:
                chunks.append(" ".join(current))

            return chunks

    # Last resort: hard token split
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    return [
        tokenizer.decode(
            ids[i:i + max_tokens],
            skip_special_tokens=True,
        )
        for i in range(0, len(ids), max_tokens)
    ]


def get_context_chunks(context: str) -> list[str]:
    if token_len(context) <= CHUNK_SIZE:
        return [context]

    sentences = sent_tokenize(context)

    chunks = []
    current = []

    for sentence in sentences:
        sentence_parts = split_oversized_segment(
            sentence,
            CHUNK_SIZE,
        )

        for part in sentence_parts:
            candidate = " ".join(current + [part])

            if token_len(candidate) <= CHUNK_SIZE:
                current.append(part)
            else:
                if current:
                    chunks.append(" ".join(current))

                current = [part]

    if current:
        chunks.append(" ".join(current))

    print(
        "Chunked token lengths:",
        [token_len(chunk) for chunk in chunks],
    )

    return chunks