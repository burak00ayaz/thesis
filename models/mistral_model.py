import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MistralModel:
    def __init__(self):
        MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,      # torch_dtype is deprecated
            device_map="auto"
        )
        
    def answer_question(self, question: str, context: str, max_new_tokens: int = 128) -> str:
        prompt = f"""
    You are a question-answering assistant.

    Answer the question using only the provided context.
    If the answer is not contained in the context, say:
    "I don't know based on the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """.strip()

        messages = [
            {"role": "user", "content": prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return answer.strip()


