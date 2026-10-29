import gc
from pathlib import Path
import torch
from transformers import AutoTokenizer
import shutil


from models.model import ModelAbstract
from corpus.types import Triplet
from models.xRAG.src.model import SFR, XMistralForCausalLM
from models.xRAG.src.language_modeling.utils import XRAG_TOKEN, get_retrieval_embeds


class xRAGModel(ModelAbstract):
    def __init__(self):
        super().__init__(
            backbone_model="Mistral-7B-Instruct-v0.2",
            soft_compression="xrag-7b",
            compression_ratio=128,
        )

        self.device = "cuda"
        self.dtype = torch.float16

        self.xrag_name = "Hannibal046/xrag-7b"
        self.retriever_name = "Salesforce/SFR-Embedding-Mistral"

        # Change this to wherever you keep experiment artifacts.
        self.compression_dir = Path("models/xRAG/temp")
        self.compression_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embedding_path(self, triplet: Triplet) -> Path:
        """
        Gives every (dataset, question_id) pair its own compressed-context file.
        """
        dataset_name = (
            triplet.dataset.value
            if hasattr(triplet.dataset, "value")
            else str(triplet.dataset)
        )

        dataset_dir = self.compression_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        return dataset_dir / f"{triplet.question_id}.pt"

    def _unload_model(self, model=None):
        if model is not None:
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _remove_temp_files(self):
        if self.compression_dir.exists():
            shutil.rmtree(self.compression_dir)

    # ------------------------------------------------------------------
    # 1. Compression
    # ------------------------------------------------------------------

    def batch_compress_and_save(self, triplets: list[Triplet]):
        """
        Compress all context chunks using SFR-Embedding-Mistral and save the
        resulting retrieval embeddings to disk.

        Each context chunk corresponds to exactly one xRAG embedding.
        """

        retriever_tokenizer = AutoTokenizer.from_pretrained(
            self.retriever_name
        )

        retriever = SFR.from_pretrained(
            self.retriever_name,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        for i, triplet in enumerate(triplets):
            if not triplet.context:
                raise ValueError(
                    f"Triplet {triplet.question_id} has an empty context."
                )
            print(f"Compressing context for {triplet.question_id} (item {i+1}/{len(triplets)})...")
            # context is list[str], therefore this creates a batch:
            #
            # [
            #   chunk_1 tokens,
            #   chunk_2 tokens,
            #   ...
            # ]
            retriever_inputs = retriever_tokenizer(
                triplet.context,
                max_length=180, # todo should not have this hardcoded.
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                retrieval_embeds = get_retrieval_embeds(
                    retriever,
                    retriever_inputs["input_ids"],
                    retriever_inputs["attention_mask"],
                )

            # Important:
            # retrieval_embeds shape should be approximately:
            #
            # [number_of_context_chunks, embedding_dimension]
            #
            # Move to CPU before saving so the file does not depend on CUDA.
            retrieval_embeds = retrieval_embeds.cpu()

            if retrieval_embeds.shape[0] != len(triplet.context):
                raise RuntimeError(
                    f"Embedding/chunk mismatch for "
                    f"{triplet.question_id}: "
                    f"{retrieval_embeds.shape[0]} embeddings for "
                    f"{len(triplet.context)} context chunks."
                )

            torch.save(
                {
                    "retrieval_embeds": retrieval_embeds,
                    "num_chunks": len(triplet.context),
                },
                self._embedding_path(triplet),
            )

        # SFR is large. Remove it before loading xRAG.
        self._unload_model(retriever)

    # ------------------------------------------------------------------
    # 2. Load compressed contexts and answer
    # ------------------------------------------------------------------

    def batch_load_and_answer(self, triplets: list[Triplet], max_new_tokens: int = 128) -> list[str]:
        """
        Load previously compressed context embeddings and generate an answer
        for every Triplet.

        Returns:
            list[str]: generated answers in the same order as triplets.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            self.xrag_name,
            padding_side="left",
            add_eos_token=False,
            use_fast=False,
        )

        model = XMistralForCausalLM.from_pretrained(
            self.xrag_name,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

        xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)

        model.set_xrag_token_id(xrag_token_id)

        answers = []

        for i, triplet in enumerate(triplets):
            print(f"Processing question {i+1}/{len(triplets)}: {triplet.question_id}")
            embedding_path = self._embedding_path(triplet)

            if not embedding_path.exists():
                raise FileNotFoundError(
                    f"No compressed context found for "
                    f"{triplet.question_id}: {embedding_path}"
                )

            saved = torch.load(
                embedding_path,
                map_location="cpu",
            )

            retrieval_embeds = saved["retrieval_embeds"]
            num_chunks = saved["num_chunks"]

            if num_chunks != len(triplet.context):
                raise RuntimeError(
                    f"Saved xRAG embedding count does not match current "
                    f"context for {triplet.question_id}. "
                    f"Saved: {num_chunks}, "
                    f"current: {len(triplet.context)}."
                )

            if retrieval_embeds.shape[0] != num_chunks:
                raise RuntimeError(
                    f"Corrupted xRAG embedding file for "
                    f"{triplet.question_id}: "
                    f"{retrieval_embeds.shape[0]} embeddings but "
                    f"metadata says {num_chunks} chunks."
                )

            # One <xRAG> token per compressed context chunk.
            compressed_context = "\n".join(
                f"Context {i + 1}: {XRAG_TOKEN}"
                for i in range(num_chunks)
            )

            prompt = (
                "Answer the question based on the provided context.\n\n"
                f"{compressed_context}\n\n"
                f"Question: {triplet.question}\n"
            )

            prompt = f"[INST] {prompt} [/INST] Answer:"

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.device)

            # Safety check:
            # number of xRAG tokens must equal number of embeddings.
            num_xrag_tokens = (
                inputs["input_ids"] == xrag_token_id
            ).sum().item()

            if num_xrag_tokens != num_chunks:
                raise RuntimeError(
                    f"xRAG token/embedding mismatch for "
                    f"{triplet.question_id}: "
                    f"{num_xrag_tokens} xRAG tokens but "
                    f"{num_chunks} embeddings."
                )

            with torch.inference_mode():
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    retrieval_embeds=retrieval_embeds.to(
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            answer = tokenizer.decode(
                output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            answers.append(answer)

        self._unload_model(model)
        self._remove_temp_files()

        return answers

    # ------------------------------------------------------------------
    # 3. Convenience method
    # ------------------------------------------------------------------

    def answer_questions_batch(self, triplets: list[Triplet]) -> list[str]:
        """
        Compress contexts first, then load the compressed representations and
        answer all questions.
        """
        self.batch_compress_and_save(triplets)
        return self.batch_load_and_answer(triplets)

    def answer_question(self, triplet: Triplet) -> str:
        """
        Compress the context first, then load the compressed representation and
        answer the question.
        """
        self.batch_compress_and_save([triplet])
        return self.batch_load_and_answer([triplet])[0]