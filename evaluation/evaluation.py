from results_db.results_service import ResultsService, ResultEntryDB

def deliver_general_exact_match_accuracy(results: list[ResultEntryDB]):
    total = len(results)
    correct = sum(1 for r in results if r.answer_in_output)
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"EXACT MATCH ACCURACY")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")

def deliver_evaluation_by_label(results: list[ResultEntryDB]):
    label_to_results = {}
    for result in results:
        label = result.answer_label
        if label not in label_to_results:
            label_to_results[label] = []
        label_to_results[label].append(result)

    print(f"EVALUATION BY LABEL")
    for label, results_for_label in label_to_results.items():
        total = len(results_for_label)
        correct = sum(1 for r in results_for_label if r.answer_in_output)
        accuracy = correct / total if total > 0 else 0.0
        print(f"Label: {label}, Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")


def squad_validation_multi_answer_eval():
    DB_PATH = "data/squad_validation/pisco_squad_chunked.db"
    DB_OUT_PATH = "data/squad_validation/pisco_squad_chunked_multi_answer.db"
    results_service = ResultsService(DB_PATH)
    results = results_service.get_all_results()
    results_service_out = ResultsService(DB_OUT_PATH)
    
    from corpus.questions.SQUAD.squad_validation import get_by_question_id
    from corpus.types import Triplet, Dataset
    from evaluation.util import normalize_string
    from models.model import Model
    correct_count = 0
    total_count = len(results)
    
    for i, result in enumerate(results):
        print(f"Evaluating result with index {i}/{total_count}")
        model_output = result.model_output
        answers = get_by_question_id(result.question_id)["answers"]["text"]
        normalized_model_output = normalize_string(model_output)
        normalized_answers = [normalize_string(ans) for ans in answers]
        exact_match = any(ans in normalized_model_output for ans in normalized_answers)
        if exact_match:
            correct_count += 1

        # Save the result to the output database
        result_entry = ResultEntryDB(
            run_id=result.run_id,
            dataset=result.dataset,
            question_id=result.question_id,
            context=result.context,
            question=result.question,
            answer='; '.join(answers),  # Store all answers as a single string
            answer_label=result.answer_label,
            model=result.model,
            soft_compression=result.soft_compression,
            compression_ratio=result.compression_ratio,
            lexical_side_channel_method=result.lexical_side_channel_method,
            lexical_side_channel_output=result.lexical_side_channel_output,
            model_output=result.model_output,
            answer_in_output=exact_match,
            normalized_indel=None,
            indel_substring=None,
        )
        results_service_out.add_result_db(result_entry)

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"SQUAD Validation Multi-Answer Evaluation")
    print(f"Total: {total_count}, Correct: {correct_count}, Accuracy: {accuracy:.2%}")

def squad_validation_filter_token_length(): 
    DB_PATH = "data/squad_validation/pisco_squad_multi_answer.db"
    results_service = ResultsService(DB_PATH)
    results = results_service.get_all_results()

    from transformers import AutoTokenizer

    # tokenizer of the PISCO backbone model
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    def token_len(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    filtered_results = []
    accepted_range = (0, 140)  # Define the accepted token length range
    for i, result in enumerate(results):
        print(f"Processing result {i + 1}/{len(results)}", end="\r")
        context = " ".join(result.context)
        token_length = token_len(context)
        if accepted_range[0] <= token_length <= accepted_range[1]:
            filtered_results.append(result)

    # calculate accuracy for the filtered results
    total = len(filtered_results)
    correct = sum(1 for r in filtered_results if r.answer_in_output)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nFiltered EXACT MATCH ACCURACY")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
    print(f"Filtered results: {len(filtered_results)} out of {len(results)}")


def squad_entity_centric_analysis():
    DB_PATH = "data/squad_validation/pisco_squad_chunked_multi_answer.db"
    results_service = ResultsService(DB_PATH)
    results = results_service.get_all_results()
    results = results[:3000]

    def get_result_by_question_id(question_id: str) -> ResultEntryDB:
        for result in results:
            if result.question_id == question_id:
                return result
        return None  # Return None if no result is found for the given question_id

    import json
    json_path = "corpus/filtering/entity_questions/squad_validation.json"
    with open(json_path, "r") as f:
        entity_questions = json.load(f)

    entity_question_results = {}

    for question_id, entity_question in entity_questions.items():
        result = get_result_by_question_id(question_id)

        if result is not None:
            entity_question_results[question_id] = {
                "correct_answer": result.answer_in_output,
                "entity_question": entity_question,
            }
        
    # analyze by entity type
    entity_type_to_results = {}
    for question_id, data in entity_question_results.items():
        entity_type = data["entity_question"]["label"]
        if entity_type not in entity_type_to_results:
            entity_type_to_results[entity_type] = []
        entity_type_to_results[entity_type].append(data["correct_answer"])

    # general accuracy
    total = len(entity_question_results)
    correct = sum(1 for data in entity_question_results.values() if data["correct_answer"])
    accuracy = correct / total if total > 0 else 0.0
    print(f"General Accuracy for Entity-Centric Questions:")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")
        
    print(f"Entity-Centric Analysis:")
    for entity_type, correctness_list in entity_type_to_results.items():
        total = len(correctness_list)
        correct = sum(1 for is_correct in correctness_list if is_correct)
        accuracy = correct / total if total > 0 else 0.0
        print(f"Entity Type: {entity_type}, Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}")


def pisco_chunks_length_analysis():
    DB_PATH = "data/squad_validation/pisco_squad_chunked.db"
    results_service = ResultsService(DB_PATH)
    results = results_service.get_all_results()

    from transformers import AutoTokenizer

    # tokenizer of the PISCO backbone model
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2"
    )

    def token_len(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    total_token_count = 0
    total_context_chunk_count = 0
    token_lengths = []

    for i, result in enumerate(results):
        for context_chunk in result.context:
            token_length = token_len(context_chunk)
            print(f"Token length: {token_length}")
            total_token_count += token_length
            total_context_chunk_count += 1
            token_lengths.append(token_length)

    average_token_length = total_token_count / total_context_chunk_count if total_context_chunk_count > 0 else 0.0
    print(f"Average Token Length for PISCO Chunks: {average_token_length:.2f}")

def squad_pisco_popularity_analysis():
    DB_PATH = "data/squad_validation/pisco_squad_chunked_multi_answer.db"
    results_service = ResultsService(DB_PATH)
    results = results_service.get_all_results()

    from corpus.questions.SQUAD.squad_validation import get_by_question_id
    import json
    json_path = "evaluation/wikilinker/entity_inlink_popularity.json"

    with open(json_path, "r") as f:
        inlink_popularities = json.load(f)

    popularity_scores = []
    wiki_resolved_count = 0
    wiki_resolved_results = []

    for i, result in enumerate(results):
        if result.question_id in inlink_popularities.keys():
            wiki_resolved_results.append({
                "result": result,
                "popularity": inlink_popularities[result.question_id]["popularity"]
            })



import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def squad_pisco_popularity_analysis_gpt(is_correct_fn, n_bins=10):
    DB_PATH = "data/squad_validation/mistral_squad_multi_answer.db"
    JSON_PATH = "evaluation/wikilinker/entity_inlink_popularity.json"

    results_service = ResultsService(DB_PATH)
    results = results_service.get_all_results()

    with open(JSON_PATH, "r") as f:
        inlink_popularities = json.load(f)

    # Join evaluation results with entity popularity
    rows = []

    for result in results:
        entity_data = inlink_popularities.get(result.question_id)
        if not entity_data:
            continue

        popularity = entity_data.get("popularity")
        incoming_links = entity_data.get("incoming_links")

        if popularity is None or incoming_links is None:
            continue

        rows.append({
            "question_id": result.question_id,
            "popularity": float(popularity),
            "incoming_links": int(incoming_links),
            "correct": int(is_correct_fn(result)),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("No questions with valid popularity scores.")
        return None

    # Quantile-based popularity buckets
    df["bucket"] = pd.qcut(
        df["popularity"],
        q=n_bins,
        duplicates="drop",
    )

    summary = (
        df.groupby("bucket", observed=True)
        .agg(
            count=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
            median_links=("incoming_links", "median"),
            min_links=("incoming_links", "min"),
            max_links=("incoming_links", "max"),
        )
        .reset_index()
    )

    summary["accuracy_pct"] = summary["accuracy"] * 100

    # Wilson 95% confidence intervals
    z = 1.96
    p = summary["accuracy"].to_numpy()
    n = summary["count"].to_numpy()

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = (
        z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        / denominator
    )

    lower = (center - margin) * 100
    upper = (center + margin) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(summary))
    y = summary["accuracy_pct"].to_numpy()

    ax.errorbar(
        x, y,
        yerr=[y - lower, upper - y],
        fmt="o-",
        capsize=4,
        linewidth=2,
        markersize=6,
    )

    labels = [
        f"{int(row.min_links):,}–{int(row.max_links):,}"
        for row in summary.itertuples()
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Entity popularity (incoming Wikipedia links)")
    ax.set_ylabel("PISCO answer accuracy (%)")
    ax.set_title("PISCO Accuracy by Answer Entity Popularity")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    output_path = Path(
        "evaluation/wikilinker/mistral_accuracy_vs_popularity.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Total evaluated questions: {len(results)}")
    print(f"Popularity-resolved questions: {len(df)}")
    print(f"Coverage: {len(df) / len(results):.1%}")
    print(f"Overall resolved accuracy: {df['correct'].mean():.1%}")

    print(
        summary[
            ["count", "correct", "accuracy_pct",
             "min_links", "max_links"]
        ].to_string(index=False)
    )

    return summary



if __name__ == "__main__":
    squad_pisco_popularity_analysis_gpt(
        is_correct_fn=lambda r: r.answer_in_output,
        n_bins=10,
    )
    exit(0)
    RESULTS_DB_PATHS = {
        "nq": [
            "data/entity_questions/mistral_nq.db",
            "data/entity_questions/pisco_nq.db",
            "data/entity_questions/xrag_nq.db",
        ],
        "bioasq": [
            "data/entity_questions/mistral_bioasq.db",
            "data/entity_questions/pisco_bioasq.db",
            "data/entity_questions/xrag_bioasq.db",
        ],
        "squad": [
            "data/entity_questions/mistral_squad.db",
            "data/entity_questions/pisco_squad.db",
            "data/entity_questions/xrag_squad.db",
        ],
        "squad_validation": [
            "data/squad_validation/pisco_squad_multi_answer.db",
            "data/squad_validation/mistral_squad_multi_answer.db",
            "data/squad_validation/pisco_squad_chunked_multi_answer.db",
        ]
    }
    for dataset, db_paths in RESULTS_DB_PATHS.items():
        print("=" * 50)
        print(f"{dataset.upper()} DATASET RESULTS")
        for db_path in db_paths:
            print("-" * 50)
            print(f"Evaluating results from database: {db_path}")
            results_service = ResultsService(db_path)
            results = results_service.get_all_results()
            results = results[:3000]  # Limit to first 3000 results for evaluation
            deliver_general_exact_match_accuracy(results)
            deliver_evaluation_by_label(results)
        print("=" * 50)
        print("\n\n")