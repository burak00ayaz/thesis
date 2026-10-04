from questions.types import Dataset, Triplet
from questions.questions import triplets
from lexical_side_channel.lsc import LexicalSideChannelBaseline, get_lexical_side_channel_method
# from models.mistral_model import MistralModel
from models.PISCO.pisco_model import PiscoModel
from logger import log

spacy_ner = get_lexical_side_channel_method(LexicalSideChannelBaseline.SPACY_NER)
pisco_model = PiscoModel()

for i, (question_real, question_synth) in enumerate(zip(triplets(Dataset.SQUAD), triplets(Dataset.SQUAD_MAPPED))):

    print(f"Example {i} Real")
    print(f"Question: {question_real.question}")
    print(f"Context: {question_real.context}")
    print(f"Answer: {question_real.answer}")

    side_channel_output = spacy_ner.process(question_real.context)
    query_lsc = side_channel_output + " " + question_real.question
    output = pisco_model.answer_question(question_real.question, question_real.context)
    output_lsc = pisco_model.answer_question(query_lsc, question_real.context)

    print(f"Side channel output: {side_channel_output}")
    print(f"Model output: {output}")
    print(f"Model output with LSC: {output_lsc}")
    print(80 * "=")

    print(f"Example {i} Synth")
    print(f"Question: {question_synth.question}")
    print(f"Context: {question_synth.context}")
    print(f"Answer: {question_synth.answer}")

    side_channel_output = spacy_ner.process(question_synth.context)
    query_lsc = side_channel_output + " " + question_synth.question
    output = pisco_model.answer_question(question_synth.question, question_synth.context)
    output_lsc = pisco_model.answer_question(query_lsc, question_synth.context)

    print(f"Side channel output: {side_channel_output}")
    print(f"Model output: {output}")
    print(f"Model output with LSC: {output_lsc}")
    print(80 * "=")

    if i == 10:
        break


