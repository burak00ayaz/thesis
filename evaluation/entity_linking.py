from results_db.results_service import ResultsService, ResultEntryDB
from evaluation.wikidata_linker import WikidataLinker


RESULTS_DB_PATH = "data/mistral_nq.db"

results_service = ResultsService(RESULTS_DB_PATH)
results = results_service.get_all_results()
linker = WikidataLinker()


for i, result in enumerate(results):
    if result.answer_label != "PERSON":
        continue

    candidates = linker.link(result.answer)

    print(80 * "-")
    print(f"\nAnswer: {result.answer}")
    print(f"Model Output: {result.model_output}")
    print(f"Context: {result.context}")
    print(f"Question: {result.question}")

    for candidate in candidates:
        print(
            candidate["id"],
            candidate.get("label"),
            candidate.get("description"),
        )
    if i == 4:
        break
