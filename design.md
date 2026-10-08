# id: str -> primary key
# run_id: str -> refers to the experiment id, a specific run (can also be named run_id)

# dataset: str (Dataset Enum)
# question_id: str -> id of the Triplet object. Real-Mapped pair of questions will have the same ID. 
#                     (dataset, question_id) pair will be the natural identity of the question. 

# context_json: list[str]
# question: str
# answer: str

# model: str -> Backbone model. "Mistral-7B-Instruct-v0.2"
# soft_compression: str -> NULL, "PISCO", "xRAG"
# compression_rate: float -> we might want to test PISCO with different compr. rates.
# lexical_side_channel_method: str -> NULL, "spacy_lsc"

# model_output: str
# answer_in_output: int

# created_at: Date

Let's think about the requirements. I want to be able to 

- calculate the accuracy of a (model,lsc,database,run_id) 
- for a given real question, find the imaginary pair and vice versa