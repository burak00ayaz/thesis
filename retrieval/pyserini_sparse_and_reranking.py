import json
from dataclasses import dataclass
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers.cross_encoder import CrossEncoder


@dataclass
class RetrievedPassage:
    rank_bm25: int
    docid: str
    bm25_score: float
    text: str
    rerank_score: float | None = None


def extract_text(raw: str) -> str:
    """
    Pyserini raw documents are often JSON strings.
    This helper extracts a useful passage text field.
    """
    if raw is None:
        return ""

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    for key in ["contents", "text", "body", "passage"]:
        if key in obj and obj[key]:
            return obj[key]

    return raw


def sparse_retrieve(
    searcher: LuceneSearcher,
    question: str,
    bm25_top_k: int = 50,
) -> list[RetrievedPassage]:
    hits = searcher.search(question, k=bm25_top_k)

    passages = []

    for rank, hit in enumerate(hits, start=1):
        doc = searcher.doc(hit.docid)
        raw = doc.raw() if doc is not None else ""
        text = extract_text(raw)

        if not text.strip():
            continue

        passages.append(
            RetrievedPassage(
                rank_bm25=rank,
                docid=hit.docid,
                bm25_score=hit.score,
                text=text,
            )
        )

    return passages


def rerank(
    reranker: CrossEncoder,
    question: str,
    passages: list[RetrievedPassage],
) -> list[RetrievedPassage]:
    pairs = [[question, p.text] for p in passages]

    scores = reranker.predict(pairs)

    for passage, score in zip(passages, scores):
        passage.rerank_score = float(score)

    return sorted(
        passages,
        key=lambda p: p.rerank_score,
        reverse=True,
    )


# Sparse BM25 retrieval over Wikipedia DPR 100-word passages.
searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")
searcher.set_bm25(k1=0.9, b=0.4)

# Small MS MARCO cross-encoder reranker.
# This downloads a model, but not a huge retrieval index.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def retrieve_and_rerank(question: str, answer: str, bm25_top_k: int = 50, final_top_k: int = 5) -> list[RetrievedPassage]:
    candidates = sparse_retrieve(
        searcher=searcher,
        question=question,
        bm25_top_k=bm25_top_k,
    )

    # filter out the candidates that do not contain the answer string (case-insensitive)
    candidates = [
        p for p in candidates if answer.lower() in p.text.lower()
    ]

    reranked = rerank(
        reranker=reranker,
        question=question,
        passages=candidates,
    )

    return reranked[:final_top_k]


def main():
    question = "Who performed In Violet Light?"
    answer = "The Tragically Hip"

    reranked_passages = retrieve_and_rerank(question, answer)

    print(f"Question: {question}")
    # print(f"BM25 candidates: {len(candidates)}")
    print(f"Final reranked top-k: {final_top_k}")
    print()

    for new_rank, passage in enumerate(reranked_passages, start=1):
        print(f"Reranked rank: {new_rank}")
        print(f"Original BM25 rank: {passage.rank_bm25}")
        print(f"DocID: {passage.docid}")
        print(f"BM25 score: {passage.bm25_score:.4f}")
        print(f"Reranker score: {passage.rerank_score:.4f}")
        print(passage.text[:800].replace("\n", " "))
        print("-" * 100)


if __name__ == "__main__":
    main()