import json
import sqlite3
import uuid

from datetime import datetime, timezone
from pathlib import Path


class ResultsDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._create_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,

                    dataset TEXT NOT NULL,
                    question_id TEXT NOT NULL,

                    context_json TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,

                    model TEXT NOT NULL,
                    soft_compression TEXT,
                    compression_ratio REAL,
                    lexical_side_channel_method TEXT,
                    lexical_side_channel_output TEXT,

                    model_output TEXT NOT NULL,
                    answer_in_output INTEGER NOT NULL
                        CHECK (answer_in_output IN (0, 1)),

                    created_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_config
                ON results (
                    run_id,
                    dataset,
                    model,
                    lexical_side_channel_method
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_question
                ON results (
                    dataset,
                    question_id
                )
                """
            )

    def insert(
        self,
        *,
        run_id: str,
        dataset: str,
        question_id: str,
        context: list[str],
        question: str,
        answer: str,
        model: str,
        soft_compression: str | None,
        compression_ratio: float | None,
        lexical_side_channel_method: str | None,
        lexical_side_channel_output: str | None,
        model_output: str,
        answer_in_output: bool,
    ) -> str:
        result_id = str(uuid.uuid4())

        created_at = datetime.now(timezone.utc).isoformat()

        context_json = json.dumps(
            context,
            ensure_ascii=False,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO results (
                    id,
                    run_id,
                    dataset,
                    question_id,
                    context_json,
                    question,
                    answer,
                    model,
                    soft_compression,
                    compression_ratio,
                    lexical_side_channel_method,
                    lexical_side_channel_output,
                    model_output,
                    answer_in_output,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    run_id,
                    dataset,
                    question_id,
                    context_json,
                    question,
                    answer,
                    model,
                    soft_compression,
                    compression_ratio,
                    lexical_side_channel_method,
                    lexical_side_channel_output,
                    model_output,
                    int(answer_in_output),
                    created_at,
                ),
            )

        return result_id

    def fetch_one(
        self,
        result_id: str,
    ) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT *
                FROM results
                WHERE id = ?
                """,
                (result_id,),
            ).fetchone()

    def fetch_all(
        self,
        *,
        filters: dict[str, object] | None = None,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM results"

        parameters = []

        if filters:
            conditions = []

            for column, value in filters.items():
                if value is None:
                    conditions.append(f"{column} IS NULL")
                else:
                    conditions.append(f"{column} = ?")
                    parameters.append(value)

            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at"

        with self._connect() as conn:
            return conn.execute(
                query,
                parameters,
            ).fetchall()

    def delete(self, result_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM results
                WHERE id = ?
                """,
                (result_id,),
            )

        return cursor.rowcount > 0