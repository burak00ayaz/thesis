from questions.types import Dataset, Triplet
from questions.questions import triplets
from models.model import Model, get_model
from lexical_side_channel.lsc import LexicalSideChannelBaseline, get_lexical_side_channel_method
from results_db.results_service import ResultsService, ResultEntry

def run_experiment(
    run_id: str,
    dataset: Dataset,
    count: int | None,
    model_enum: Model,
    lexical_side_channel_method_enum: LexicalSideChannelBaseline | None,
    results_database_path: str,
):
    results_service = ResultsService(results_database_path)
    model = get_model(model_enum)
    triplets_dataset = triplets(dataset)


    if lexical_side_channel_method_enum is not None:
        lsc_method = get_lexical_side_channel_method(lexical_side_channel_method_enum)

    for i, triplet in enumerate(triplets_dataset):

        if count is not None and i >= count:
            break

        print(f"Example {i}/{count if count is not None else "All"}")

        input_question = triplet.question

        if lexical_side_channel_method_enum is not None:
            side_channel = lsc_method.process(triplet.context)
            input_question = f"{side_channel} {triplet.question}"

        output = model.answer_question(input_question, triplet.context)
        answer_correctness = triplet.answer.lower() in output.lower()

        result_entry = ResultEntry(
            run_id=run_id,
            triplet=triplet,
            model=model,
            lexical_side_channel_method=lexical_side_channel_method_enum.value if lexical_side_channel_method_enum is not None else None,
            lexical_side_channel_output=side_channel if lexical_side_channel_method_enum is not None else None,
            model_output=output,
            answer_in_output=answer_correctness
        )
        results_service.add_result(result_entry)