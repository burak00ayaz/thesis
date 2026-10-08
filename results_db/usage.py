from results_db.results_service import ResultsService, ResultEntryDB
from questions.types import Dataset

def main():
    service = ResultsService("data/example_results.db")

    result = ResultEntryDB(
        run_id="pisco_16x_spacy_001",

        dataset=Dataset.SQUAD_MAPPED,
        question_id="5733be284776f41900661180",

        context=[
            "Barack Obama was born in Hawaii.",
            "He served as the 44th president of the United States.",
        ],

        question="Where was Barack Obama born?",
        answer="Hawaii",

        model="Mistral-7B-Instruct-v0.2",

        soft_compression="PISCO",
        compression_ratio=16.0,
        lexical_side_channel_method="spacy_lsc",

        model_output="Barack Obama was born in Hawaii.",
        answer_in_output=True,
    )

    # ------------------------------------------------------------------
    # Add result
    # ------------------------------------------------------------------

    result_id = service.add_result_db(result)

    print("Inserted result:")
    print(result_id)
    print()

    # ------------------------------------------------------------------
    # Get result by ID
    # ------------------------------------------------------------------

    stored_result = service.get_result(result_id)

    print("Stored result:")
    print(stored_result)
    print()

    # ------------------------------------------------------------------
    # Calculate accuracy
    # ------------------------------------------------------------------

    accuracy = service.get_accuracy(
        run_id="pisco_16x_spacy_001",
        dataset=Dataset.SQUAD_MAPPED,
        model="Mistral-7B-Instruct-v0.2",
        lexical_side_channel_method="spacy_lsc",
        soft_compression="PISCO",
        compression_ratio=16.0,
    )

    print("Accuracy:")
    print(accuracy)
    print()

    # ------------------------------------------------------------------
    # Get statistics
    # ------------------------------------------------------------------

    statistics = service.get_statistics(
        run_id="pisco_16x_spacy_001",
        dataset=Dataset.SQUAD_MAPPED,
        model="Mistral-7B-Instruct-v0.2",
        lexical_side_channel_method="spacy_lsc",
    )

    print("Statistics:")
    print(statistics)
    print()

    # ------------------------------------------------------------------
    # Find real / mapped counterpart
    # ------------------------------------------------------------------

    pair = service.find_pair(
        dataset=Dataset.SQUAD,
        question_id="5733be284776f41900661180",
    )

    print("Paired question results:")
    print(pair)


if __name__ == "__main__":
    main()