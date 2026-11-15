from lexical_side_channel.lsc import LexicalSideChannelMethod
from tf_idf.process_kilt import score_text

class KiltLSC(LexicalSideChannelMethod):

    """
    A tf-idf based lexical side channel method using term frequencies from the KILT dataset.
    """
    def process(self, context: list[str]) -> str:
        context_text = " ".join(context)
        scores = score_text(
            (context_text),
            db_path="tf_idf/kilt21M.sqlite"
        )
        # return top 5 terms with highest tf-idf scores
        top_terms = sorted(scores, key=lambda x: x["idf"], reverse=True)
        top_terms = [term["term"] for term in top_terms[:5]]
        return f"({'|'.join(top_terms)})"