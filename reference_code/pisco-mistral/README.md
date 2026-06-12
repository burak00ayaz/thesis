---
license: cc-by-4.0
language:
- en
base_model:
- mistralai/Mistral-7B-Instruct-v0.2
---

# Model Card for PISCO-mistral


PISCO is a context compression model to be used for efficient inference when doing Retrieval Augmented Generation (RAG), particularly optimized for question answering.

PISCO contains two adapters around a backbone LLM:
- An encoder adapter trained to perform compression of input contexts (the retrieved documents in RAG) into a set of 8 embedding vectors
- A decoder adapter, which can take as input sets of embeddings vectors from documents and a query and provide an answer

With a compressed collection of documents to retrieve from, inference becomes about x5 faster. PISCO models have very small loss in accuracy on a wide set of QA benchmarks (0-3%).

*Developed by*: Naver Labs Europe  
*License*: [CC BY-NC 4.0.](https://creativecommons.org/licenses/by-nc/4.0/)  
* *Model*: `Pisco-mistral` 
* *Backbone model*: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
* *Model size*: 7.33 billion parameters
* *Compression rate*: x16: each document (of size up to 128 tokens) is converted into 8 embedding vectors.

## Usage

```python
from transformers import AutoModel

pisco = AutoModel.from_pretrained('naver/pisco-mistral').to('cuda')

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
out = pisco.generate_from_text(questions=questions, documents=documents, max_new_tokens=64)
print('Generated answer', out)

# Document compression:
embeddings = pisco.compress_documents(documents=documents[0])

# Generation from compressed documents:
out = pisco.generate_from_compressed_documents_and_questions(questions=questions, compressed_documents=embeddings)
``` 

The recommended usage is to provide documents cropped to about 128 tokens, which is common practice when doing RAG.

## Model features

* **PISCO enables high accuracy responses from the compressed documents**
* **PISCO is robust to various domains** We tested its compression/decoding abilities on various sets of data.
* **PISCO enables x5 faster generation** when the collection documents to retrieve from is pre-compressed.

## License

This work is licensed under CC BY-NC 4.0. 

## Cite

```
@article{louis2025pisco,
  title={Pisco: Pretty simple compression for retrieval-augmented generation},
  author={Louis, Maxime and D{\'e}jean, Herv{\'e} and Clinchant, St{\'e}phane},
  journal={arXiv preprint arXiv:2501.16075},
  year={2025}
}

```

## Acknowledgements

Model trained at [Naver Labs Europe](https://europe.naverlabs.com/)  
Team:
* [Maxime LOUIS](https://europe.naverlabs.com/people_user_naverlabs/maxime-louis/)
* [Hervé Dejean](https://europe.naverlabs.com/people_user_naverlabs/herve-dejean/)
* [Stéphane Clinchant](https://europe.naverlabs.com/people_user_naverlabs/st%C3%A9phane-clinchant/)