from transformers import AutoModel
import hashlib

class PiscoModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            'naver/pisco-mistral',
            trust_remote_code=True
        ).to('cuda')
        self.current_compressed_context_hash = ""
        self.embeddings = None

    def hash_context(self, context: str) -> str:
        return hashlib.sha256(context.encode()).hexdigest()

    def answer_question(self, question: str, context: str, max_new_tokens: int = 128) -> str:
        question_array = [question]

        # End-to-end usage
        # out = self.model.generate_from_text(questions=question_array, documents=[documents], max_new_tokens=max_new_tokens)

        # Document compression:
        context_hash = self.hash_context(context)
        if context_hash != self.current_compressed_context_hash:
            self.embeddings = self.model.compress_documents(documents=[context])
            self.current_compressed_context_hash = context_hash

        # Generation from compressed documents:
        out = self.model.generate_from_compressed_documents_and_questions(questions=question_array, compressed_documents=self.embeddings)
        return out[0]