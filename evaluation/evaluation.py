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


def eval_numerical_results():
    PATHS = {
        "squad": [
            "data/numerical/pisco.db",
            "data/numerical/pisco_lsc.db",
            "data/numerical/xrag.db",
            "data/numerical/xrag_lsc.db",
        ]
    }
    for dataset, db_paths in PATHS.items():
        print("=" * 50)
        print(f"{dataset.upper()} DATASET RESULTS")
        for db_path in db_paths:
            print("-" * 50)
            print(f"Evaluating results from database: {db_path}")
            results_service = ResultsService(db_path)
            results = results_service.get_all_results()
            deliver_general_exact_match_accuracy(results)
            deliver_evaluation_by_label(results)
        print("=" * 50)
        print("\n\n")

def eval_ranked_ner_results():
    PATH = {
        "squad": [
            "data/ranked_ner/pisco_ranked_ner_filtered.db",
            "data/ranked_ner/pisco_ranked_ner.db",
            "data/ranked_ner/pisco_vanilla.db",
        ]
    }
    for dataset, db_paths in PATH.items():
        print("=" * 50)
        print(f"{dataset.upper()} DATASET RESULTS")
        for db_path in db_paths:
            print("-" * 50)
            print(f"Evaluating results from database: {db_path}")
            results_service = ResultsService(db_path)
            results = results_service.get_all_results()
            deliver_general_exact_match_accuracy(results)
            deliver_evaluation_by_label(results)
        print("=" * 50)
        print("\n\n")


if __name__ == "__main__":
    eval_ranked_ner_results()
    exit(0)
    RESULTS_DB_PATHS = {
        "nq": [
            "data/mistral_nq.db",
            "data/pisco_nq.db",
            "data/xrag_nq.db",
        ],
        "bioasq": [
            "data/mistral_bioasq.db",
            "data/pisco_bioasq.db",
            "data/xrag_bioasq.db",
        ],
        "squad": [
            "data/mistral_squad.db",
            "data/pisco_squad.db",
            "data/xrag_squad.db",
        ],
    }
    for dataset, db_paths in RESULTS_DB_PATHS.items():
        print("=" * 50)
        print(f"{dataset.upper()} DATASET RESULTS")
        for db_path in db_paths:
            print("-" * 50)
            print(f"Evaluating results from database: {db_path}")
            results_service = ResultsService(db_path)
            results = results_service.get_all_results()
            deliver_general_exact_match_accuracy(results)
            deliver_evaluation_by_label(results)
        print("=" * 50)
        print("\n\n")