from datasets import load_dataset
qa_rm = load_dataset("kenqgu/SynthWorlds", "qa-rm", split="test")
qa_sm = load_dataset("kenqgu/SynthWorlds", "qa-sm", split="test")


def get_question_context_answer(example):
    return {
        "question": example["query"], 
        "context": example["gold_docs"], 
        "answer": example["gold_answers"]
    }


for i in range(5):
    real_example = get_question_context_answer(qa_rm[i])
    img_example = get_question_context_answer(qa_sm[i])
    print('\n\nReal example: ', real_example)
    print('\n\nImaginary example: ', img_example)