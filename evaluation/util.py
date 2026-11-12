import re
import string


def normalize_string(text: str) -> str:
    """Normalize a string according to PISCO's evaluation procedure."""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Remove articles ("a", "an", "the")
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # 4. Standardize whitespace
    text = " ".join(text.split())

    return text
