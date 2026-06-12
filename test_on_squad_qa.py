from models.mistral_model import answer_question as mistral_answer_question
from models.pisco_model import answer_question as pisco_answer_question
from questions.squad import get_question_context_answer_triples
from logger import log


for i, triple in enumerate(get_question_context_answer_triples()):
    log(f"Example {i}:")
    log("Question:", triple["question"])
    log("Context:", triple["context"])
    log("Answer:", triple["answer"])
    log("-------------------------------")

    out_mistral = mistral_answer_question(triple["question"], triple["context"])
    log("Mistral Answer:", out_mistral)
    out_pisco = pisco_answer_question(triple["question"], triple["context"])
    log("Pisco Answer:", out_pisco)
    log("-------------------------------")