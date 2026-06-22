#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def clean_line(line: str) -> str:
    match = re.match(r"^\d{4}-\d{2}-\d{2} .*? \| INFO \| ?(.*)$", line.rstrip("\n"))
    return match.group(1) if match else line.rstrip("\n")


def parse_log(path: Path, model: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    current = None
    collecting_answer = False
    answer_lines: list[str] = []
    capture_next_context = False
    model_title = model.capitalize()

    def flush_answer() -> None:
        nonlocal collecting_answer, answer_lines
        if current is not None and collecting_answer:
            current[f"{model}_answer"] = "\n".join(answer_lines).strip()
        collecting_answer = False
        answer_lines = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = clean_line(raw_line)

        example_match = re.match(r"^Example (\d+)\s*$", line)
        if example_match:
            flush_answer()
            current = {"example": int(example_match.group(1))}
            rows[current["example"]] = current
            capture_next_context = False
            continue

        if current is None:
            continue

        if collecting_answer:
            verdict_match = re.match(
                rf"^{model_title} answer (contains|does NOT contain) the correct answer string\.",
                line,
            )
            if (
                verdict_match
                or line.startswith("=")
            ):
                flush_answer()
                if verdict_match:
                    current[f"{model}_correct"] = verdict_match.group(1) == "contains"
            else:
                answer_lines.append(line)
                continue

        if capture_next_context:
            if line and not line.startswith("-"):
                current["retrieved_context"] = line
            capture_next_context = False
            continue

        if line.startswith("Question: "):
            current["question"] = line[len("Question: ") :]
        elif line.startswith("Answer: "):
            current["answer"] = line[len("Answer: ") :]
        elif line.startswith("Answer Entity Popularity: "):
            value = line[len("Answer Entity Popularity: ") :]
            current["answer_popularity"] = int(float(value))
        elif line.startswith("Rerank score: "):
            # The retrieved passage is the next INFO line after the rerank score.
            # Some passages start with a quoted title, but not all of them do.
            capture_next_context = True
        else:
            answer_match = re.match(rf"^{model_title} Answer: (.*)$", line)
            if answer_match:
                collecting_answer = True
                answer_lines = [answer_match.group(1)]

    flush_answer()
    return rows


def merge_answered_rows(mistral_rows: dict[int, dict], pisco_rows: dict[int, dict]) -> list[dict]:
    merged = []
    examples = sorted(set(mistral_rows) | set(pisco_rows))

    for example in examples:
        mistral = mistral_rows.get(example, {})
        pisco = pisco_rows.get(example, {})
        mistral_answer = mistral.get("mistral_answer", "")
        pisco_answer = pisco.get("pisco_answer", "")

        # Keep only rows where at least one model generated an answer.
        if not mistral_answer and not pisco_answer:
            continue

        merged.append(
            {
                "example": example,
                "question": mistral.get("question") or pisco.get("question", ""),
                "answer": mistral.get("answer") or pisco.get("answer", ""),
                "mistral_answer": mistral_answer,
                "mistral_correct": mistral.get("mistral_correct", ""),
                "pisco_answer": pisco_answer,
                "pisco_correct": pisco.get("pisco_correct", ""),
                "retrieved_context": mistral.get("retrieved_context")
                or pisco.get("retrieved_context", ""),
                "answer_popularity": mistral.get("answer_popularity")
                or pisco.get("answer_popularity", ""),
            }
        )

    return merged


def write_csv(rows: list[dict], path: Path) -> None:
    columns = [
        "example",
        "question",
        "answer",
        "mistral_answer",
        "mistral_correct",
        "pisco_answer",
        "pisco_correct",
        "retrieved_context",
        "answer_popularity",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx_if_available(rows: list[dict], path: Path) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False

    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Answered Questions", index=False)
        summary = pd.DataFrame(
            [
                {"metric": "answered_rows", "value": len(rows)},
                {
                    "metric": "missing_context_rows",
                    "value": sum(1 for row in rows if not row["retrieved_context"]),
                },
            ]
        )
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistral-log", required=True, type=Path)
    parser.add_argument("--pisco-log", required=True, type=Path)
    parser.add_argument("--out-prefix", default="granola_answered_questions", type=Path)
    args = parser.parse_args()

    mistral_rows = parse_log(args.mistral_log, "mistral")
    pisco_rows = parse_log(args.pisco_log, "pisco")
    rows = merge_answered_rows(mistral_rows, pisco_rows)

    csv_path = args.out_prefix.with_suffix(".csv")
    xlsx_path = args.out_prefix.with_suffix(".xlsx")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(rows, csv_path)
    wrote_xlsx = write_xlsx_if_available(rows, xlsx_path)

    print(f"mistral_log_examples: {len(mistral_rows)}")
    print(f"pisco_log_examples: {len(pisco_rows)}")
    print(f"answered_rows: {len(rows)}")
    print(f"missing_context_rows: {sum(1 for row in rows if not row['retrieved_context'])}")
    print(f"csv: {csv_path}")
    if wrote_xlsx:
        print(f"xlsx: {xlsx_path}")
    else:
        print("xlsx: skipped, install pandas and openpyxl to enable XLSX export")


if __name__ == "__main__":
    main()
