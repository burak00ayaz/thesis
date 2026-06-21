from datasets import load_dataset


class GranolaEntityQuestions:
    def __init__(self, relations: list[str] | None = None, sort_by_answer_popularity: bool = False, answer_popularity_threshold: int | None = None):
        # Login using e.g. `huggingface-cli login` to access this dataset
        self.ds = load_dataset("google/granola-entity-questions", token=True)

        # print(type(self.ds))
        # <class 'datasets.dataset_dict.DatasetDict'>

        # print(self.ds["train"].features)
        # {'Unnamed: 0': Value('int64'), 'relation': Value('string'), 'question': Value('string'), 'question_entity': Value('string'), 'question_entity_qid': Value('string'), 'question_entity_description': Value('string'), 'question_entity_popularity': Value('float64'), 'answer': Value('string'), 'answer_entity_qid': Value('string'), 'answer_entity_description': Value('string'), 'answer_entity_popularity': Value('float64'), 'score_for_potential_error': Value('float64'), 'granola_answer_1': Value('string'), 'granola_answer_2': Value('string'), 'granola_answer_3': Value('string'), 'granola_answer_4': Value('string'), 'granola_answer_5': Value('string'), 'granola_answer_6': Value('string'), 'granola_answer_7': Value('string'), 'granola_answer_8': Value('string'), 'granola_answer_9': Value('string'), 'granola_answer_10': Value('string'), 'granola_answer_11': Value('float64'), 'granola_answer_12': Value('float64'), 'granola_answer_13': Value('float64'), 'granola_answer_14': Value('float64'), 'num_granola_answers': Value('int64')}

        # print(len(self.ds["train"]))
        # 12452

        if relations is not None:
            self.ds["train"] = self.ds["train"].filter(
                lambda x: x["relation"] in relations
            )
        # Length of P26 samples: 899

        # sort the dataset by answer entity popularity in ascending order
        if sort_by_answer_popularity:
            self.ds["train"] = self.ds["train"].sort(column_names="answer_entity_popularity", reverse=False)

        # filter out examples where the answer entity popularity is below a certain threshold
        self.ds["train"] = self.ds["train"].filter(
            lambda x: x["answer_entity_popularity"] is not None
            and x["answer_entity_popularity"] >= answer_popularity_threshold
        )

    def get_question_answer_tuples(self):
        for i in range(len(self.ds["train"])):
            example = self.ds["train"][i]
            question = example["question"]
            answer = example["answer"]
            answer_entity_popularity = example["answer_entity_popularity"]
            yield {"question": question, "answer": answer, "answer_entity_popularity": answer_entity_popularity}


if __name__ == "__main__":
    granola = GranolaEntityQuestions(relations=["P26"], sort_by_answer_popularity=True, answer_popularity_threshold=100)
    for i in range(10):
        example = granola.ds["train"][i]
        question = example["question"]
        answer = example["answer"]
        answer_entity_qid = example["answer_entity_qid"]
        answer_entity_description = example["answer_entity_description"]
        answer_entity_popularity = example["answer_entity_popularity"]

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Answer Entity QID: {answer_entity_qid}")
        print(f"Answer Entity Description: {answer_entity_description}")
        print(f"Answer Entity Popularity: {answer_entity_popularity}")
        print("-" * 50) 