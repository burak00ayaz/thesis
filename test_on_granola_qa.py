from questions.granola_entity_questions import GranolaEntityQuestions
from retrieval.pyserini_sparse_and_reranking import retrieve_and_rerank
from logger import log
from models.mistral_model import MistralModel
from models.pisco_model import PiscoModel

granola = GranolaEntityQuestions(relations=["P26"], sort_by_answer_popularity=True, answer_popularity_threshold=100)

def test_mistral_on_granola_qa():
    mistral_model = MistralModel()
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

        out_mistral = mistral_model.answer_question(question, reranked_passages[0].text)
        log(f"Mistral Answer: {out_mistral}")
        
        # check if the answer string is contained in the Mistral answer (case-insensitive)
        if answer.lower() in out_mistral.lower():
            log("Mistral answer contains the correct answer string.")
            correct_answer_count += 1
        else:
            log("Mistral answer does NOT contain the correct answer string.")
        
        log("=" * 80)
        total_count += 1

        if i >= 100:
            break

    log(f"Correct answers: {correct_answer_count}/{total_count}")


def test_pisco_on_granola_qa():
    pisco_model = PiscoModel()
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

        out_pisco = pisco_model.answer_question(question, reranked_passages[0].text)
        log(f"Pisco Answer: {out_pisco}")
        
        # check if the answer string is contained in the Pisco answer (case-insensitive)
        if answer.lower() in out_pisco.lower():
            log("Pisco answer contains the correct answer string.")
            correct_answer_count += 1
        else:
            log("Pisco answer does NOT contain the correct answer string.")
        
        log("=" * 80)
        total_count += 1

        if i >= 100:
            break

    log(f"Correct answers: {correct_answer_count}/{total_count}")


test_pisco_on_granola_qa()