from questions.questions import Dataset, triplets
from lexical_side_channel.lsc import LexicalSideChannelBaseline, get_lexical_side_channel_method
# from models.mistral_model import MistralModel
from models.pisco_model import PiscoModel
from logger import log

spacy_ner = get_lexical_side_channel_method(LexicalSideChannelBaseline.SPACY_NER)
pisco_model = PiscoModel()

for i, triplet in enumerate(triplets(Dataset.SQUAD_MAPPED)):
    if i <= 30:
        continue

    side_channel_output = spacy_ner.process(triplet.context)
    query_lsc = side_channel_output + " " + triplet.question

    print(f"Example {i}")
    print(f"Question: {triplet.question}")
    print(f"Context: {triplet.context}")
    print(f"Answer: {triplet.answer}")
    print(f"Side channel output: {side_channel_output}")

    output = pisco_model.answer_question(triplet.question, triplet.context)
    output_lsc = pisco_model.answer_question(query_lsc, triplet.context)

    print(f"Model output: {output}")
    print(f"Model output with LSC: {output_lsc}")
    print(80 * "=")

    if i == 60:
        break


