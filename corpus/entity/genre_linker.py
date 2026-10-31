from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pathlib import Path
import json

class GenreLinker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink").eval()

    def link(self, context: str, answer_entity: str, start_char: int, end_char: int):
        marked_context = (
            context[: start_char]
            + "[START_ENT] "
            + answer_entity
            + " [END_ENT]"
            + context[end_char :]
        )
        outputs = self.model.generate(
            **self.tokenizer([marked_context], return_tensors="pt"),
            num_beams=5,
            num_return_sequences=5,
            max_new_tokens=32,
            return_dict_in_generate=True,
            output_scores=True,
        )

        predictions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        scores = outputs.sequences_scores
        relative_probs = torch.softmax(scores, dim=0)

        results = []
        for prediction, score, prob in zip(predictions, scores, relative_probs):
            results.append(
                {
                    "prediction": prediction,
                    "score": score.item(),
                    "probability": prob.item(),
                }
            )
        return results


# read a json file that is located under parent/filtering/entity_questions
with open(Path(__file__).parent.parent / "filtering" / "entity_questions" / "nq_2.json", "r") as f:
    entity_questions = json.load(f)

genre_linker = GenreLinker()

for question_id, entity_question in entity_questions.items():
    print(f"Question ID: {question_id}")
    print(f"Context: {' '.join(entity_question['context'])}")
    print(f"Question: {entity_question['question']}")
    print(f"Answer Entity: {entity_question['answer_entity']}")
    print(f"Label: {entity_question['label']}")
    print(80 * "-")
    results = genre_linker.link(
        context=" ".join(entity_question["context"]),
        answer_entity=entity_question["answer_entity"],
        start_char=entity_question["start"],
        end_char=entity_question["end"],
    )
    for result in results:
        print(
            f"Prediction: {result['prediction']} | "
            f"Score: {result['score']:.4f} | "
            f"Probability: {result['probability']:.2%}"
        )
    print(80 * "=")

exit(0)