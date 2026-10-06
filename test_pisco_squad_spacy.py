from questions.types import Dataset, Triplet
from questions.questions import triplets
from lexical_side_channel.lsc import LexicalSideChannelBaseline, get_lexical_side_channel_method
from models.model import get_model, Model
from results_db.results_service import ResultsService, ResultEntry

RUN_ID = "pisco_squad_spacy_001"
SKIP_GENERATION = False

results_service = ResultsService("data/bok.db")

if not SKIP_GENERATION:
    spacy_ner = get_lexical_side_channel_method(LexicalSideChannelBaseline.SPACY_NER)
    pisco_model = get_model(Model.PISCO)

    for i, (question_real, question_synth) in enumerate(zip(triplets(Dataset.SQUAD), triplets(Dataset.SQUAD_MAPPED))):
        if SKIP_GENERATION:
            break

        print(f"Example {i} Real")

        side_channel_output = spacy_ner.process(question_real.context)
        query_lsc = side_channel_output + " " + question_real.question
        output = pisco_model.answer_question(question_real.question, question_real.context)
        output_lsc = pisco_model.answer_question(query_lsc, question_real.context)

        result_entry = ResultEntry(
            run_id=RUN_ID,
            triplet=question_real,
            model=pisco_model,
            lexical_side_channel_method=None,
            model_output=output,
            answer_in_output=(question_real.answer.lower() in output.lower())
        )
        results_service.add_result(result_entry)

        result_entry_lsc = ResultEntry(
            run_id=RUN_ID,
            triplet=question_real,
            model=pisco_model,
            lexical_side_channel_method=LexicalSideChannelBaseline.SPACY_NER.value,
            model_output=output_lsc,
            answer_in_output=(question_real.answer.lower() in output_lsc.lower())
        )
        results_service.add_result(result_entry_lsc)

        print(f"Example {i} Synth")

        side_channel_output = spacy_ner.process(question_synth.context)
        query_lsc = side_channel_output + " " + question_synth.question
        output = pisco_model.answer_question(question_synth.question, question_synth.context)
        output_lsc = pisco_model.answer_question(query_lsc, question_synth.context)

        result_entry = ResultEntry(
            run_id=RUN_ID,
            triplet=question_synth,
            model=pisco_model,
            lexical_side_channel_method=None,
            model_output=output,
            answer_in_output=(question_synth.answer.lower() in output.lower())
        )
        results_service.add_result(result_entry)

        result_entry_lsc = ResultEntry(
            run_id=RUN_ID,
            triplet=question_synth,
            model=pisco_model,
            lexical_side_channel_method=LexicalSideChannelBaseline.SPACY_NER.value,
            model_output=output_lsc,
            answer_in_output=(question_synth.answer.lower() in output_lsc.lower())
        )
        results_service.add_result(result_entry_lsc)

        if i == 0:
            break

accuracy = results_service.get_accuracy(
    run_id=RUN_ID,
    dataset=Dataset.SQUAD,
    model="Mistral-7B-Instruct-v0.2",
    lexical_side_channel_method=None,
    soft_compression="PISCO",
)

print(f"Accuracy SQUAD Real: {accuracy}")

accuracy = results_service.get_accuracy(
    run_id=RUN_ID,
    dataset=Dataset.SQUAD,
    model="Mistral-7B-Instruct-v0.2",
    lexical_side_channel_method=LexicalSideChannelBaseline.SPACY_NER.value,
    soft_compression="PISCO",
)

print(f"Accuracy SQUAD Real with LSC: {accuracy}")

accuracy = results_service.get_accuracy(
    run_id=RUN_ID,
    dataset=Dataset.SQUAD_MAPPED,
    model="Mistral-7B-Instruct-v0.2",
    lexical_side_channel_method=None,
    soft_compression="PISCO",
)

print(f"Accuracy SQUAD Synth: {accuracy}")

accuracy = results_service.get_accuracy(
    run_id=RUN_ID,
    dataset=Dataset.SQUAD_MAPPED,
    model="Mistral-7B-Instruct-v0.2",
    lexical_side_channel_method=LexicalSideChannelBaseline.SPACY_NER.value,
    soft_compression="PISCO",
)

print(f"Accuracy SQUAD Synth with LSC: {accuracy}")

# Accuracy SQUAD Real: 0.7398373983739838
# Accuracy SQUAD Real with LSC: 0.7967479674796748
# Accuracy SQUAD Synth: 0.6016260162601627
# Accuracy SQUAD Synth with LSC: 0.6910569105691057