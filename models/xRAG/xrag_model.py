import gc
import torch
from transformers import AutoTokenizer
from models.xRAG.src.model import SFR, XMistralForCausalLM
from models.xRAG.src.language_modeling.utils import XRAG_TOKEN, get_retrieval_embeds
from models.model import ModelAbstract
from questions.types import Triplet


class xRAGModel(ModelAbstract):
    def __init__(self):
        super().__init__(
            backbone_model="Mistral-7B-Instruct-v0.2",
            soft_compression="xrag-7b",
            compression_ratio=128
        )
        self.device = "cuda"
        self.dtype = torch.float16
        self.xrag_name = "Hannibal046/xrag-7b"
        self.retriever_name = "Salesforce/SFR-Embedding-Mistral"

    def batch_compress_and_save(self, triplets: list[Triplet]):
        pass

    def batch_load_and_answer(self, triplets: list[Triplet]):
        pass

    def batch_answer_questions(self, triplets: list[Triplet]):
        pass

device = "cuda"
dtype = torch.float16

xrag_name = "Hannibal046/xrag-7b"
retriever_name = "Salesforce/SFR-Embedding-Mistral"

context = ["Alice lives in Madrid.", "Her favorite color is purple."]
question = "Where does Alice live?"


# --------------------------------------------------
# 1. Load retriever and compress the context
# --------------------------------------------------

retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)

retriever = SFR.from_pretrained(
    retriever_name,
    torch_dtype=dtype,
).to(device).eval()

retriever_inputs = retriever_tokenizer(
    context,
    max_length=180,
    truncation=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    retrieval_embeds = get_retrieval_embeds(
        retriever,
        retriever_inputs["input_ids"],
        retriever_inputs["attention_mask"],
    )

# Keep the tiny embedding on CPU
retrieval_embeds = retrieval_embeds.cpu()

print("retrieval_embeds:", retrieval_embeds.shape)


# --------------------------------------------------
# 2. Remove retriever from GPU
# --------------------------------------------------

del retriever
del retriever_inputs

gc.collect()
torch.cuda.empty_cache()


# --------------------------------------------------
# 3. Load xRAG
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    xrag_name,
    padding_side="left",
    add_eos_token=False,
    use_fast=False,
)

model = XMistralForCausalLM.from_pretrained(
    xrag_name,
    torch_dtype=dtype,
).to(device).eval()

model.set_xrag_token_id(
    tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
)


# --------------------------------------------------
# 4. Ask question using compressed context
# --------------------------------------------------

prompt = (
    "Answer the question based on the provided context.\n\n"
    f"Context: {XRAG_TOKEN}\n\n"
    f"Question: {question}\n"
)
prompt = f"[INST] {prompt} [/INST] Answer:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        retrieval_embeds=retrieval_embeds.to(device),
        max_new_tokens=32,
        do_sample=False,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))