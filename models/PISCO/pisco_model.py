from transformers import AutoModel
import torch
from pathlib import Path
from models.model import ModelAbstract
from corpus.types import Triplet
import shutil

class PiscoModel(ModelAbstract):
    def __init__(self):
        super().__init__(
            backbone_model="Mistral-7B-Instruct-v0.2",
            soft_compression="PISCO",
            compression_ratio=16
        )
        self.model = AutoModel.from_pretrained(
            'naver/pisco-mistral',
            trust_remote_code=True
        ).to('cuda')
        self.model.eval()

        self.compression_dir = Path("models/PISCO/temp")
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

    def _remove_temp_files(self):
        if self.compression_dir.exists():
            shutil.rmtree(self.compression_dir)

    # ------------------------------------------------------------------
    # 1. Compression
    # ------------------------------------------------------------------

    def batch_compress_and_save(self, triplets: list[Triplet]):

        for i, triplet in enumerate(triplets):
            if not triplet.context:
                raise ValueError(
                    f"Triplet {triplet.question_id} has an empty context."
                )
            print(f"Compressing context for {triplet.question_id} (item {i+1}/{len(triplets)})...")

            with torch.inference_mode():
                retrieval_embeds = self.model.compress_documents(triplet.context)

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


            with torch.inference_mode():
                output = self.model.generate_from_compressed_documents_and_questions(questions=[triplet.question], compressed_documents=retrieval_embeds)

            answers.append(output[0])

        self._remove_temp_files()
        return answers

    def answer_questions_batch(self, triplets: list[Triplet]) -> list[str]:
        self.batch_compress_and_save(triplets)
        return self.batch_load_and_answer(triplets)

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