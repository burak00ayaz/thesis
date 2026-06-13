from transformers import AutoModel

oscar = AutoModel.from_pretrained('naver/oscar-mistral-7B', trust_remote_code=True).to('cuda')

# Example documents and question:
documents = [
    [
        "Weldenia is a monotypic genus of flowering plant in the family Commelinaceae, first describ ed in 1829. It has one single species: Weldenia candida, which grows originally in Mexico and Guatemala.",
        "Hagsatera is a genus of flowering plants from the orchid family, Orchidaceae. There are two known species, native to Mexico and Guatemala",
        "Alsobia is a genus of flowering plants in the family Gesneriaceae, native to Mexico, Guatemala and Costa Rica. The two species are succulent, stoloniferous herbs and were previously included in the genus \"Episcia\". Recent molecular studies have supported the separation of \"Alsobia\" from \"Episcia\""
    ]
]

questions = ["Which genus of plant grows originally in Mexico and Guatemala, Phylica or Weldenia?"]

# End-to-end usage
out = oscar.generate_from_text(questions=questions, documents=documents, max_new_tokens=64, query_dependent=True)
print('Generated answer', out)

# Document compression:
embeddings = oscar.compress_documents(documents=documents[0], questions=questions * len(documents[0])) # compression is query-dependent, one question per doc here

# Generation from compressed documents:
out = oscar.generate_from_compressed_documents_and_questions(questions=questions, compressed_documents=embeddings)
print('Generated answer from compressed documents:', out)