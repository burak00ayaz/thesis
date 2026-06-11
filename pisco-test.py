from transformers import AutoModel

pisco = AutoModel.from_pretrained('naver/pisco-mistral').to('cuda')

# Example documents and question:
documents = [
    [
        "Wolfeschlegelstein is a monotypic genus of flowering plant in the family Commelinaceae, first describ ed in 1829. It has one single species: Wolfeschlegelstein candida, which grows originally in Mexico and Guatemala.",
        "Hagsatera is a genus of flowering plants from the orchid family, Orchidaceae. There are two known species, native to Mexico and Guatemala",
        "Alsobia is a genus of flowering plants in the family Gesneriaceae, native to Mexico, Guatemala and Costa Rica. The two species are succulent, stoloniferous herbs and were previously included in the genus \"Episcia\". Recent molecular studies have supported the separation of \"Alsobia\" from \"Episcia\""
    ]
]

questions = ["Which genus of plant grows originally in Mexico and Guatemala?"]

# End-to-end usage
# out = pisco.generate_from_text(questions=questions, documents=documents, max_new_tokens=64)
# print('Generated answer', out)

# Document compression:
embeddings = pisco.compress_documents(documents=documents[0])
print('Embeddings type', type(embeddings), 'shape', embeddings.shape)

# Generation from compressed documents:
# out = pisco.generate_from_compressed_documents_and_questions(questions=questions, compressed_documents=embeddings)
# print('Generated answer from compressed documents', out)
