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


def deliver_evaluation_by_popularity(results: list[ResultEntryDB]):
    import json
    from pathlib import Path

    popularity_results_path = (
        Path(__file__).parent.parent
        / "nq_google_stuff"
        / "popularity_results.json"
    )

    with open(popularity_results_path, "r", encoding="utf-8") as f:
        popularity_results = json.load(f)

    buckets = [
        (0, 500),
        (500, 1_000),
        (1_000, 5_000),
        (5_000, 10_000),
        (10_000, 50_000),
        (50_000, 100_000),
        (100_000, float("inf")),
    ]

    bucket_results = {bucket: [] for bucket in buckets}
    missing_popularity = 0

    for result in results:
        popularity_score = popularity_results.get(
            result.question_id, {}
        ).get("popularity")

        if popularity_score is None:
            missing_popularity += 1
            continue

        popularity_score = float(popularity_score)

        for lower, upper in buckets:
            if lower <= popularity_score < upper:
                bucket_results[(lower, upper)].append(result)
                break

    print("\nAccuracy by popularity bucket")
    print("-" * 60)
    print(f"{'Popularity':<22} {'Samples':>10} {'Accuracy':>12}")
    print("-" * 60)

    evaluation = {}

    for (lower, upper), entries in bucket_results.items():
        label = (
            f"{lower:,}+"
            if upper == float("inf")
            else f"{lower:,}–{upper:,}"
        )

        total = len(entries)
        correct = sum(entry.answer_in_output for entry in entries)
        accuracy = correct / total if total else None

        evaluation[label] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
        }

        accuracy_str = (
            f"{accuracy:.2%}" if accuracy is not None else "N/A"
        )

        print(f"{label:<22} {total:>10} {accuracy_str:>12}")

    print("-" * 60)
    print(f"Missing popularity: {missing_popularity}")

    return evaluation

        




# if __name__ == "__main__":
#     RESULTS_DB_PATHS = {
#         "nq": [
#             "data/mistral_nq.db",
#             "data/pisco_nq.db",
#             "data/xrag_nq.db",
#         ],
#         "bioasq": [
#             "data/mistral_bioasq.db",
#             "data/pisco_bioasq.db",
#             "data/xrag_bioasq.db",
#         ],
#         "squad": [
#             "data/mistral_squad.db",
#             "data/pisco_squad.db",
#             "data/xrag_squad.db",
#         ],
#     }
#     for dataset, db_paths in RESULTS_DB_PATHS.items():
#         print("=" * 50)
#         print(f"{dataset.upper()} DATASET RESULTS")
#         for db_path in db_paths:
#             print("-" * 50)
#             print(f"Evaluating results from database: {db_path}")
#             results_service = ResultsService(db_path)
#             results = results_service.get_all_results()
#             deliver_general_exact_match_accuracy(results)
#             deliver_evaluation_by_label(results)
#         print("=" * 50)
#         print("\n\n")

if __name__ == "__main__":
    RESULTS_DB_PATH = "data/pisco_nq_google.db"

    results_service = ResultsService(RESULTS_DB_PATH)
    results = results_service.get_all_results()
    deliver_general_exact_match_accuracy(results)
    deliver_evaluation_by_label(results)
    deliver_evaluation_by_popularity(results)