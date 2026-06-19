import csv
import gzip
import json
import re
from pathlib import Path
from collections import defaultdict

import ahocorasick
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EQ_ROOT = PROJECT_ROOT / "questions" / "entity_questions"
TEMPLATES_PATH = EQ_ROOT / "relation_query_templates.json"
DPR_PATH = PROJECT_ROOT / "retrieval" / "data" / "dpr" / "psgs_w100.tsv.gz"
OUT_PATH = PROJECT_ROOT / "data" / "entityquestions_silver" / "test_with_contexts.jsonl"


STOPWORDS = {
    "what", "where", "when", "who", "which", "was", "were", "is", "are",
    "the", "of", "in", "to", "by", "for", "a", "an", "did", "does", "do",
    "has", "have", "had"
}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def get_question(ex: dict) -> str:
    for key in ["question", "query"]:
        if key in ex:
            return ex[key]
    raise KeyError(f"Could not find question key in: {ex.keys()}")


def get_answers(ex: dict) -> list[str]:
    for key in ["answers", "answer"]:
        if key in ex:
            value = ex[key]
            if isinstance(value, list):
                return [str(x) for x in value]
            return [str(value)]
    raise KeyError(f"Could not find answer key in: {ex.keys()}")


def template_to_regex(template: str) -> re.Pattern:
    """
    Converts e.g.:
      'Where was [X] born?'
    into:
      r'^Where\\ was\\ (.+?)\\ born\\?$'
    """
    escaped = re.escape(template)
    escaped = escaped.replace(re.escape("[X]"), r"(.+?)")
    return re.compile("^" + escaped + "$", re.IGNORECASE)


def extract_subject(question: str, relation: str, templates: dict) -> str | None:
    template = templates.get(relation)
    if not template:
        return None

    # Some template files may store list or string
    if isinstance(template, list):
        template = template[0]

    pattern = template_to_regex(template)
    match = pattern.match(question)
    if match:
        return match.group(1).strip()

    return None


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def boundary_ok(text: str, start: int, end: int) -> bool:
    """
    Prevents matching 'York' inside weird larger strings.
    start/end are Python slice positions.
    """
    before_ok = start == 0 or not text[start - 1].isalnum()
    after_ok = end == len(text) or not text[end].isalnum()
    return before_ok and after_ok


def load_entityquestions(split="test") -> list[dict]:
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        templates = json.load(f)

    examples = []

    for path in sorted((EQ_ROOT / split).glob(f"P*.{split}.json")):
        relation = path.name.split(".")[0]

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, ex in enumerate(data):
            question = get_question(ex)
            answers = get_answers(ex)
            subject = ex.get("subject") or ex.get("entity") or extract_subject(
                question, relation, templates
            )

            examples.append({
                "id": f"{relation}:{idx}",
                "relation": relation,
                "question": question,
                "answers": answers,
                "subject": subject,
                "original": ex,
            })

    return examples


def build_answer_automaton(examples: list[dict]):
    """
    Maps answer strings to example IDs and builds fast multi-string matcher.
    """
    answer_to_example_ids = defaultdict(list)

    for i, ex in enumerate(examples):
        for answer in ex["answers"]:
            answer_norm = normalize_text(answer)
            if len(answer_norm) >= 2:
                answer_to_example_ids[answer_norm].append(i)

    automaton = ahocorasick.Automaton()

    for answer_norm in answer_to_example_ids:
        automaton.add_word(answer_norm, answer_norm)

    automaton.make_automaton()
    return automaton, answer_to_example_ids


def score_candidate(ex: dict, passage: dict) -> int:
    question = ex["question"]
    subject = ex.get("subject")

    text = passage["text"]
    title = passage["title"]

    text_norm = normalize_text(text)
    title_norm = normalize_text(title)

    score = 0

    # Strong signal: subject appears exactly
    if subject:
        subject_norm = normalize_text(subject)

        if subject_norm == title_norm:
            score += 20
        elif subject_norm in title_norm:
            score += 12
        elif subject_norm in text_norm:
            score += 10

        # Partial subject-token overlap
        subject_terms = content_words(subject)
        passage_terms = content_words(title + " " + text)
        score += 2 * len(subject_terms & passage_terms)

    # Weak signal: question words overlap with passage
    q_terms = content_words(question)
    p_terms = content_words(title + " " + text)
    score += len(q_terms & p_terms)

    return score


def build_silver_contexts():
    examples = load_entityquestions("test")
    print(f"Loaded {len(examples)} EntityQuestions examples")

    automaton, answer_to_example_ids = build_answer_automaton(examples)

    best = {
        i: {
            "score": -1,
            "passage": None,
        }
        for i in range(len(examples))
    }

    with gzip.open(DPR_PATH, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in tqdm(reader, desc="Scanning DPR passages"):
            passage_id = row["id"]
            title = row["title"]
            text = row["text"]

            searchable = normalize_text(title + " " + text)

            seen_answers = set()

            for end_idx, answer_norm in automaton.iter(searchable):
                start_idx = end_idx - len(answer_norm) + 1
                end_slice = end_idx + 1

                if not boundary_ok(searchable, start_idx, end_slice):
                    continue

                if answer_norm in seen_answers:
                    continue

                seen_answers.add(answer_norm)

                passage = {
                    "id": passage_id,
                    "title": title,
                    "text": text,
                    "matched_answer": answer_norm,
                }

                for ex_id in answer_to_example_ids[answer_norm]:
                    ex = examples[ex_id]
                    score = score_candidate(ex, passage)

                    if score > best[ex_id]["score"]:
                        best[ex_id] = {
                            "score": score,
                            "passage": passage,
                        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    matched = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i, ex in enumerate(examples):
            passage = best[i]["passage"]

            if passage is not None:
                matched += 1

            output = {
                "id": ex["id"],
                "relation": ex["relation"],
                "question": ex["question"],
                "answers": ex["answers"],
                "subject": ex["subject"],
                "silver_context": passage["text"] if passage else None,
                "silver_context_title": passage["title"] if passage else None,
                "silver_context_passage_id": passage["id"] if passage else None,
                "matched_answer": passage["matched_answer"] if passage else None,
                "silver_score": best[i]["score"],
                "context_source": "dpr_wikipedia_answer_containing",
                "original": ex["original"],
            }

            out.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"Matched {matched}/{len(examples)} examples")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build_silver_contexts()