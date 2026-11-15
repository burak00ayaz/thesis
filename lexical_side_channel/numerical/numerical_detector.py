"""Extract numerical expressions from text chunks using only Python's stdlib.

Designed for lexical side channels: preserves exact source text and offsets.
This is a high-recall heuristic detector, not a complete mathematical parser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable
from lexical_side_channel.lsc import LexicalSideChannelMethod


@dataclass(frozen=True)
class NumericalMention:
    chunk_index: int
    start: int  # character offset in the original chunk (inclusive)
    end: int    # character offset in the original chunk (exclusive)
    text: str   # exact source span, including recognized unit if present
    value_text: str  # expression without the trailing unit
    category: str
    unit: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# Longer patterns are processed before shorter patterns to avoid breaking up
# dates, percentages and ranges into isolated numbers.
NUM = r"[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?(?:st|nd|rd|th)?"
MAG = r"(?:thousand|million|billion|trillion)"
QUANTITY = rf"{NUM}(?:\s+{MAG})?"
MONTH = (r"(?:January|February|March|April|May|June|July|August|"
         r"September|October|November|December|Jan\.?|Feb\.?|Mar\.?|"
         r"Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|"
         r"Oct\.?|Nov\.?|Dec\.?)")

IDENTIFIER_RX = re.compile(
    r"\b[A-Za-z][A-Za-z0-9-]*\d[A-Za-z0-9-]*\b"
)

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("identifier", IDENTIFIER_RX),
    ("date", re.compile(rf"(?<!\w)(?:{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}\s+\d{{4}}|{MONTH}\s+\d{{4}})\b", re.I)),
    ("date", re.compile(r"(?<!\w)(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})(?!\w)")),
    ("decade", re.compile(r"(?<!\w)(?:(?:early|mid|late)[- ]+)?(?:1\d{3}|20\d0)s\b", re.I)),
    ("century", re.compile(r"(?<!\w)\d{1,2}(?:st|nd|rd|th)[- ]+century\b", re.I)),
    ("range", re.compile(rf"(?<!\w)between\s+{QUANTITY}\s+and\s+{QUANTITY}(?!\w)", re.I)),
    ("range", re.compile(rf"(?<!\w)(?:from\s+)?{QUANTITY}\s*(?:–|—|\bto\b|-(?=\d))\s*{QUANTITY}(?!\w)", re.I)),
    ("percentage", re.compile(rf"(?<!\w){QUANTITY}\s*(?:%|percent(?:age points)?\b|per\s+cent\b)", re.I)),
    ("money", re.compile(rf"(?<!\w)(?:[$€£]\s*{QUANTITY}|{QUANTITY}\s+(?:USD|EUR|GBP|dollars?|euros?|pounds?))(?!\w)", re.I)),
    ("fraction", re.compile(r"(?<!\w)\d+\s*/\s*\d+(?!\w)")),
    ("ratio", re.compile(r"(?<!\w)\d+\s*:\s*\d+(?!\w)")),
    ("number", re.compile(rf"(?<!\w)(?<![A-Za-z]-){QUANTITY}(?!\w)", re.I)),
]

# Optional adjacent units. Keep exact spelling ("tons" vs "tonnes", etc.).
UNIT_RX = re.compile(
    r"\s+(?P<unit>(?:metric\s+)?(?:tonnes?|tons?|kilograms?|grams?|kg|mg|"
    r"lbs?|pounds?|kilometers?|kilometres?|km|meters?|metres?|miles?|"
    r"centimeters?|cm|millimeters?|mm|feet|ft|inches?|lit(?:er|re)s?|"
    r"gallons?|hours?|minutes?|seconds?|days?|weeks?|months?|years?|"
    r"people|persons|dollars?|euros?|percent|per\s+cent))\b",
    re.I,
)

CARDINAL_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand "
    "million billion trillion"
).split()
ORDINAL_WORDS = (
    "first second third fourth fifth sixth seventh eighth ninth tenth "
    "eleventh twelfth thirteenth fourteenth fifteenth sixteenth "
    "seventeenth eighteenth nineteenth twentieth thirtieth fortieth "
    "fiftieth sixtieth seventieth eightieth ninetieth hundredth "
    "thousandth millionth billionth"
).split()
OTHER_WORDS = ["half", "quarter", "dozen", "twice", "thrice"]
WORD = "|".join(sorted(map(re.escape, CARDINAL_WORDS + ORDINAL_WORDS + OTHER_WORDS), key=len, reverse=True))
NUMBER_WORD_TOKEN_RX = re.compile(rf"(?<!\w)(?:{WORD})(?!\w)", re.I)
SCALE_WORDS = {"hundred", "thousand", "million", "billion", "trillion"}


def _word_number_spans(chunk: str):
    """Merge adjacent written number words without joining separate quantities.

    e.g. "one hundred and twenty" is one value, while
    "twenty and eighty" is two values.
    """
    tokens = list(NUMBER_WORD_TOKEN_RX.finditer(chunk))
    if not tokens:
        return

    start = tokens[0].start()
    end = tokens[0].end()
    prev_word = tokens[0].group().lower()

    for token in tokens[1:]:
        gap = chunk[end:token.start()]
        gap_is_joiner = gap in {" ", "-"} or (gap == " and " and prev_word in SCALE_WORDS)
        if gap_is_joiner:
            end = token.end()
            prev_word = token.group().lower()
            continue

        yield start, end
        start, end = token.start(), token.end()
        prev_word = token.group().lower()

    yield start, end


ROMAN_RX = re.compile(r"\b(?:II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XX|XXX)\b")


def _category_for_number(value_text: str) -> str:
    v = value_text.strip()
    if re.fullmatch(r"\d{4}", v) and 1000 <= int(v) <= 2099:
        return "year_or_integer"  # ambiguous without context
    if re.fullmatch(r"[+\-−]?\d+(?:st|nd|rd|th)", v, re.I):
        return "ordinal"
    return "number"


def extract_numerical_information(contexts: str | Iterable[str]) -> list[NumericalMention]:
    """Return non-overlapping mentions with exact chunk-local offsets.

    Input: one text chunk or a list of chunks (e.g. ``Triplet.context``).
    Output: ordered NumericalMention objects, one per detected expression.
    """
    if isinstance(contexts, str):
        contexts = [contexts]

    results: list[NumericalMention] = []
    for chunk_index, chunk in enumerate(contexts):
        if not isinstance(chunk, str):
            raise TypeError(f"Chunk {chunk_index} must be str, got {type(chunk).__name__}")
        taken: list[tuple[int, int]] = []
        chunk_mentions: list[NumericalMention] = []

        def add(start: int, end: int, category: str, attach_unit: bool = False) -> None:
            if any(start < existing_end and end > existing_start for existing_start, existing_end in taken):
                return
            value_text = chunk[start:end]
            unit = None
            if attach_unit:
                unit_match = UNIT_RX.match(chunk, end)
                if unit_match is not None:
                    unit = unit_match.group("unit")
                    end = unit_match.end()
            if any(start < existing_end and end > existing_start for existing_start, existing_end in taken):
                return
            taken.append((start, end))
            chunk_mentions.append(NumericalMention(
                chunk_index=chunk_index,
                start=start,
                end=end,
                text=chunk[start:end],
                value_text=value_text,
                category=category,
                unit=unit,
            ))

        for category, pattern in PATTERNS:
            for match in pattern.finditer(chunk):
                kind = _category_for_number(match.group()) if category == "number" else category
                add(match.start(), match.end(), kind, attach_unit=category in {"number", "range", "fraction", "ratio"})

        for start, end in _word_number_spans(chunk):
            value_text = chunk[start:end]
            kind = "ordinal" if re.split(r"[ -]+", value_text.lower())[-1] in ORDINAL_WORDS else "number_word"
            if re.match(r"\s+(?:percent|per\s+cent)\b", chunk[end:], re.I):
                kind = "percentage"
            add(start, end, kind, attach_unit=True)

        for match in ROMAN_RX.finditer(chunk):
            add(match.start(), match.end(), "roman_numeral")

        results.extend(sorted(chunk_mentions, key=lambda x: x.start))

    return results


class NumericalSideChannel(LexicalSideChannelMethod):
    def process(self, context: list[str]) -> str:
        context_text = " ".join(context)
        mentions = extract_numerical_information(context_text)
        side_channel = "|".join([mention.text for mention in mentions])
        return f"({side_channel})"


if __name__ == "__main__":
    example_context = [
        "According to a 2010 Pew Global Attitudes survey, 84% supported one measure, "
        "77% another and 82% a third.",
        "Eisenhower brought in 400,000 tons of food. Another figure was 731,842,320. "
        "In December 1547, there were sixteen participants and 2.2 billion people.",
    ]
    for mention in extract_numerical_information(example_context):
        print(mention.to_dict())
