from models.mistral_model import MistralModel
from models.pisco_model import PiscoModel
from questions.squad import get_question_context_answer_triples
from logger import log
from questions.squad import map_entities_in_triple

mistral_model = MistralModel()
pisco_model = PiscoModel()

for i, triple in enumerate(get_question_context_answer_triples()):
    log(f"Example {i}:")
    log("Question: ", triple["question"])
    log("Context: ", triple["context"])
    log("Answer: ", triple["answer"])

    out_mistral = mistral_model.answer_question(triple["question"], triple["context"])
    log("Mistral Answer: ", out_mistral)
    out_pisco = pisco_model.answer_question(triple["question"], triple["context"])
    log("Pisco Answer: ", out_pisco)
    log("-------------------------------")

    triple = map_entities_in_triple(triple["question"], triple["context"], triple["answer"])
    log("Mapped Question: ", triple["question"])
    log("Mapped Context: ", triple["context"])
    log("Mapped Answer: ", triple["answer"])

    out_mistral = mistral_model.answer_question(triple["question"], triple["context"])
    log("Mapped Mistral Answer: ", out_mistral)
    out_pisco = pisco_model.answer_question(triple["question"], triple["context"])
    log("Mapped Pisco Answer: ", out_pisco)
    log("===============================")