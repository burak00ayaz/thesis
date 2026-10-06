from __future__ import annotations
from dataclasses import dataclass
import json
from results_db.db import ResultsDatabase
from questions.types import Dataset, Triplet, DATASET_PAIRS
from models.model import ModelAbstract

@dataclass
class ResultEntryDB:
    run_id: str

    dataset: Dataset
    question_id: str

    context: list[str]
    question: str
    answer: str

    model: str

    soft_compression: str | None = None
    compression_ratio: float | None = None
    lexical_side_channel_method: str | None = None

    model_output: str = ""
    answer_in_output: bool = False


@dataclass
class ResultEntry:
    run_id: str
    triplet: Triplet
    model: ModelAbstract
    lexical_side_channel_method: str | None = None
    model_output: str = ""
    answer_in_output: bool = False


class ResultsService:
    def __init__(self, database_path: str):
        self.database = ResultsDatabase(database_path)

    def add_result(
        self,
        result: ResultEntry,
    ) -> str:
        return self.database.insert(
            run_id=result.run_id,
            dataset=result.triplet.dataset.value,
            question_id=result.triplet.question_id,
            context=result.triplet.context,
            question=result.triplet.question,
            answer=result.triplet.answer,
            model=result.model.backbone_model,
            soft_compression=result.model.soft_compression,
            compression_ratio=result.model.compression_ratio,
            lexical_side_channel_method=(
                result.lexical_side_channel_method
            ),
            model_output=result.model_output,
            answer_in_output=result.answer_in_output,
        )

    def add_result_db(
        self,
        result: ResultEntryDB,
    ) -> str:
        return self.database.insert(
            run_id=result.run_id,
            dataset=result.dataset.value,
            question_id=result.question_id,
            context=result.context,
            question=result.question,
            answer=result.answer,
            model=result.model,
            soft_compression=result.soft_compression,
            compression_ratio=result.compression_ratio,
            lexical_side_channel_method=(
                result.lexical_side_channel_method
            ),
            model_output=result.model_output,
            answer_in_output=result.answer_in_output,
        )

    def get_result(
        self,
        result_id: str,
    ) -> dict | None:
        row = self.database.fetch_one(result_id)

        if row is None:
            return None

        return self._row_to_dict(row)

    def get_accuracy(
        self,
        *,
        run_id: str,
        dataset: Dataset,
        model: str,
        lexical_side_channel_method: str | None,
        soft_compression: str | None = None,
        compression_ratio: float | None = None,
    ) -> float | None:
        filters = {
            "run_id": run_id,
            "dataset": dataset.value,
            "model": model,
            "lexical_side_channel_method": (
                lexical_side_channel_method
            ),
        }

        if soft_compression is not None:
            filters["soft_compression"] = soft_compression

        if compression_ratio is not None:
            filters["compression_ratio"] = compression_ratio

        rows = self.database.fetch_all(
            filters=filters,
        )

        if not rows:
            return None

        correct = sum(
            row["answer_in_output"]
            for row in rows
        )

        return correct / len(rows)

    def get_statistics(
        self,
        *,
        run_id: str,
        dataset: Dataset,
        model: str,
        lexical_side_channel_method: str | None,
    ) -> dict:
        rows = self.database.fetch_all(
            filters={
                "run_id": run_id,
                "dataset": dataset.value,
                "model": model,
                "lexical_side_channel_method": (
                    lexical_side_channel_method
                ),
            }
        )

        total = len(rows)

        if total == 0:
            return {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": None,
            }

        correct = sum(
            row["answer_in_output"]
            for row in rows
        )

        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total,
        }

    def find_pair(
        self,
        *,
        dataset: Dataset,
        question_id: str,
    ) -> list[dict]:
        paired_dataset = DATASET_PAIRS[dataset]

        rows = self.database.fetch_all(
            filters={
                "dataset": paired_dataset.value,
                "question_id": question_id,
            }
        )

        return [
            self._row_to_dict(row)
            for row in rows
        ]

    @staticmethod
    def _row_to_dict(row) -> dict:
        result = dict(row)

        result["context"] = json.loads(
            result.pop("context_json")
        )

        result["answer_in_output"] = bool(
            result["answer_in_output"]
        )

        result["dataset"] = Dataset(
            result["dataset"]
        )

        return result