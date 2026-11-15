"""Research MVP: self-information-based lexical side channel for soft RAG compression.

Install: pip install torch transformers spacy
         python -m spacy download en_core_web_sm

Example:
    import spacy
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from compression_aware_side_channel import extract_candidates, score_surprisal, select_anchors

    nlp = spacy.load('en_core_web_sm')
    lm_tok = AutoTokenizer.from_pretrained('distilgpt2', use_fast=True)
    lm = AutoModelForCausalLM.from_pretrained('distilgpt2').eval()
    spans = extract_candidates(context, nlp)
    score_surprisal(context, spans, nlp, lm, lm_tok)
    # Use the *target generator's* tokenizer to account for side-channel cost.
    anchors = select_anchors(spans, generator_tokenizer, token_budget=16,
                             predicted_loss={})

For an adaptive selector, train a model-specific loss classifier on a separate
train/validation split and supply its predicted P(soft-failure | span, context,
question, compressor) keyed by (span.start, span.end). Do not feed test gold
answers to the selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping


# Numbers, serials, years, versions; augment to cover domain-specific identifiers.
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*|\d{2,}(?:[./:-]\d+)*%?)\b"
)


@dataclass
class Candidate:
    start: int
    end: int
    text: str
    label: str
    surprisal_bits: float = 0.0
    scored_token_count: float = 0.0

    @property
    def mean_surprisal(self) -> float:
        return self.surprisal_bits / max(self.scored_token_count, 1e-9)

    @property
    def key(self) -> tuple[int, int]:
        return (self.start, self.end)


def extract_candidates(context: str, nlp, include_noun_chunks: bool = False) -> list[Candidate]:
    """Return original, verbatim surface forms and their character offsets.

    spaCy NER is a good first pass; supplementary rules catch numerical or
    alphanumeric identifiers that NER routinely misses. GLiNER can be plugged
    in here instead if already used in the evaluation framework.
    """
    doc = nlp(context)
    found: list[tuple[int, int, str]] = [
        (ent.start_char, ent.end_char, ent.label_) for ent in doc.ents
    ]
    if include_noun_chunks:
        found.extend((span.start_char, span.end_char, 'NOUN_PHRASE')
                     for span in doc.noun_chunks)
    found.extend((match.start(), match.end(), 'IDENTIFIER')
                 for match in _IDENTIFIER_RE.finditer(context))

    cleaned: list[Candidate] = []
    for start, end, label in found:
        while start < end and context[start].isspace():
            start += 1
        while start < end and context[end - 1].isspace():
            end -= 1
        if end > start:
            cleaned.append(Candidate(start, end, context[start:end], label))

    # Prefer longer exact spans over overlapping fragments such as "2019" in
    # "March 2019". Selection is still limited by its actual token budget.
    selected: list[Candidate] = []
    for span in sorted(cleaned, key=lambda c: (-(c.end - c.start), c.start)):
        if not any(span.start < kept.end and kept.start < span.end
                   for kept in selected):
            selected.append(span)
    return sorted(selected, key=lambda c: c.start)


def score_surprisal(context: str, spans: list[Candidate], nlp, lm, tokenizer) -> None:
    """In-place, sentence-local causal-LM token surprisal, in bits.

    Uses tokenizer offset mappings to assign token-level surprisal to original
    character spans. If the model/tokenizer does not insert a BOS token, the
    first token of a sentence cannot be scored and is skipped. Cross-boundary
    subword tokens are allocated proportionally to their character overlap.

    This is a reference surprisal model, NOT the PISCO/xRAG compressor.
    """
    import torch

    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Use a fast tokenizer: character offset mappings are required')
    model_device = next(lm.parameters()).device
    doc = nlp(context)
    ln2 = math.log(2.0)
    for sentence in doc.sents:
        if not any(c.start < sentence.end_char and c.end > sentence.start_char
                   for c in spans):
            continue
        encoded = tokenizer(sentence.text, return_tensors='pt',
                            return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        ids = encoded['input_ids'].to(model_device)
        if ids.shape[1] < 2:
            continue
        attention_mask = encoded.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        with torch.inference_mode():
            logits = lm(input_ids=ids, attention_mask=attention_mask).logits
            next_logprobs = logits[:, :-1].log_softmax(dim=-1)
            logp = next_logprobs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            bits = (-logp[0] / ln2).float().cpu().tolist()

        for i in range(1, len(offsets)):
            rel_start, rel_end = offsets[i]
            if rel_end <= rel_start:
                continue  # special tokens, e.g. BOS/EOS
            start = sentence.start_char + rel_start
            end = sentence.start_char + rel_end
            for span in spans:
                overlap = max(0, min(end, span.end) - max(start, span.start))
                if overlap:
                    weight = overlap / (end - start)
                    span.surprisal_bits += bits[i - 1] * weight
                    span.scored_token_count += weight


def render_anchors(spans: list[Candidate]) -> str:
    """Keep only literal surface forms; add relation-bearing labels separately
    if the generator cannot associate a name with its compressed source fact.
    """
    if not spans:
        return ''
    ordered = sorted(spans, key=lambda c: c.start)
    return 'Exact strings in the retrieved source: ' + '; '.join(s.text for s in ordered)


def select_anchors(
    spans: list[Candidate],
    generator_tokenizer,
    token_budget: int,
    predicted_loss: Mapping[tuple[int, int], float] | None = None,
    query_relevance: Mapping[tuple[int, int], float] | None = None,
    min_loss_probability: float = 0.0,
) -> list[Candidate]:
    """Greedy approximation to budgeted selection, counted with generator tokens.

    predicted_loss[(start,end)] = calibrated probability (0..1) that a span
    will be unusable for answering after compression, *when it was usable
    from full text*. If absent, this is a surprisal-only baseline, not an
    adaptive model. query_relevance is optional (0..1) and may be obtained
    by a lightweight retriever or classifier without seeing gold answers.
    """
    if token_budget < 0:
        raise ValueError('token_budget must be non-negative')
    pred = predicted_loss or {}
    rel = query_relevance or {}
    by_value: set[str] = set()
    picked: list[Candidate] = []

    def num_tokens(text: str) -> int:
        return len(generator_tokenizer.encode(text, add_special_tokens=False))

    def priority(c: Candidate) -> float:
        failure = max(0., min(1., float(pred.get(c.key, 1.))))
        relevance = max(0., min(1., float(rel.get(c.key, 1.))))
        if failure < min_loss_probability:
            return 0.
        cost = max(1, num_tokens(c.text))
        # Surprisal in bits, adjusted by model-specific failure and relevance.
        # Density avoids rewarding long noun phrases purely for their length.
        return c.surprisal_bits * failure * relevance / cost

    for span in sorted(spans, key=priority, reverse=True):
        if priority(span) <= 0 or span.text in by_value:
            continue
        proposal = picked + [span]
        if num_tokens(render_anchors(proposal)) <= token_budget:
            picked.append(span)
            by_value.add(span.text)
    return sorted(picked, key=lambda c: c.start)


if __name__ == '__main__':
    class SpaceTokenizer:
        def encode(self, s, add_special_tokens=False):
            return s.split()
    demo = [Candidate(10, 19, 'RareNameX', 'PERSON', 12., 2.),
            Candidate(45, 49, '2007', 'DATE', 9., 1.)]
    selected = select_anchors(demo, SpaceTokenizer(), 12,
                              predicted_loss={(10, 19): 0.9, (45, 49): 0.1})
    print(render_anchors(selected))
