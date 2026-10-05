from transformers import AutoModel
import torch

class PiscoModel:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            'naver/pisco-mistral',
            trust_remote_code=True
        ).to('cuda')
        self.model.eval()
        self.backbone_model = "Mistral-7B-Instruct-v0.2"
        self.soft_compression = "PISCO"
        self.compression_ratio = 16

    def answer_question(self, question: str, context: list[str], max_new_tokens: int = 128) -> str:
        question_array = [question]

        with torch.inference_mode():
            # End-to-end usage
            out = self.model.generate_from_text(questions=question_array, documents=[context], max_new_tokens=max_new_tokens)

        return out[0]

if __name__ == "__main__":
    model = PiscoModel()
    question = "What is the capital of France?"
    context = ["France is a country in Europe. Its capital city is Paris."]
    answer = model.answer_question(question, context)
    print("Answer:", answer)