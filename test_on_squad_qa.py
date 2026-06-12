from models.mistral_model import answer_question as mistral_answer_question
from models.pisco_model import answer_question as pisco_answer_question
from questions.squad import get_question_context_answer_triples


for i, triple in enumerate(get_question_context_answer_triples()):
    print(f"Example {i}:")
    print("Question:", triple["question"])
    print("Context:", triple["context"])
    print("Answer:", triple["answer"])
    print()

    out_mistral = mistral_answer_question(triple["question"], triple["context"])
    print("Mistral Answer:", out_mistral)
    out_pisco = pisco_answer_question(triple["question"], triple["context"])
    print("Pisco Answer:", out_pisco)
    print()