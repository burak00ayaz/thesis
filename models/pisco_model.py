from transformers import AutoModel

pisco = AutoModel.from_pretrained(
    'naver/pisco-mistral',
    trust_remote_code=True
).to('cuda')


def answer_question(question: str, context: str, max_new_tokens: int = 128) -> str:
    question_array = [question]
    context_array = [[context]]

    # End-to-end usage
    out = pisco.generate_from_text(questions=question_array, documents=context_array, max_new_tokens=max_new_tokens)

    # Document compression:
    # embeddings = pisco.compress_documents(documents=context_array)

    # Generation from compressed documents:
    # out = pisco.generate_from_compressed_documents_and_questions(questions=question_array, compressed_documents=embeddings)
    return out