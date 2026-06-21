from questions.GranolaEntityQuestions.granola_entity_questions import GranolaEntityQuestions
from retrieval.pyserini_sparse_and_reranking import retrieve_and_rerank
from logger import log
from models.mistral_model import MistralModel
from models.pisco_model import PiscoModel

def test_model_on_granola_qa(model_id: str):
    if model_id == "Mistral":
        model = MistralModel()
    elif model_id == "Pisco":
        model = PiscoModel()
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    correct_answer_count = 0
    total_count = 0

    for i, qa in enumerate(granola.get_question_answer_tuples()):
        question = qa["question"]
        answer = qa["answer"]
        answer_entity_popularity = qa["answer_entity_popularity"]

        log(f"Example {i+1}")
        log(f"Question: {question}")
        log(f"Answer: {answer}")
        log(f"Answer Entity Popularity: {answer_entity_popularity}")

        reranked_passages = retrieve_and_rerank(question, answer, final_top_k=1)

        if not reranked_passages:
            log("No passages retrieved that contain the answer string.")
            log("=" * 80)
            continue

        log("-" * 80)
        log(f"Reranked passages for Example {i+1}:")
        for new_rank, passage in enumerate(reranked_passages, start=1):
            log(f"Reranked rank: {new_rank}")
            log(f"Original BM25 rank: {passage.rank_bm25}")
            log(f"DocID: {passage.docid}")
            log(f"Rerank score: {passage.rerank_score:.4f}")
            log(passage.text[:500].replace("\n", " "))
            log("-" * 80)

        out = model.answer_question(question, reranked_passages[0].text)
        log(f"{model_id} Answer: {out}")
        
        # check if the answer string is contained in the {model_id} answer (case-insensitive)
        if answer.lower() in out.lower():
            log(f"{model_id} answer contains the correct answer string.")
            correct_answer_count += 1
        else:
            log(f"{model_id} answer does NOT contain the correct answer string.")
        
        log("=" * 80)
        total_count += 1

        if total_count >= 1000:
            break

    log(f"Correct answers: {correct_answer_count}/{total_count}")


granola = GranolaEntityQuestions(sort_by_answer_popularity=True)
test_model_on_granola_qa("Pisco")