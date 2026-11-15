"""Compute chunk-level KILT IDF statistics, then TF-IDF scores for text.

Install: pip install datasets
Build approximate IDF: python kilt_tfidf.py build --max-chunks 100000
Build full corpus IDF: python kilt_tfidf.py build --max-chunks 0 --db kilt_full_idf.sqlite
Score text: python kilt_tfidf.py score 'Eisenhower brought in 400,000 tons of food.'

Document = one dmrau/kilt-128 row's 'content' chunk (not a Wikipedia article).
IDF = log((1 + N)/(1 + df)) + 1 (scikit-learn smoothing).
TF-IDF uses raw term counts, with no L2 normalization.
"""

import argparse
from collections import Counter
from itertools import islice
import math
from pathlib import Path
import re
import sqlite3

# Preserve comma-separated numerals and keep common hyphenated/contracted words.
TOKEN_RE = re.compile(
    r"(?u)\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\w+(?:[-'’]\w+)*\b"
)


def tokenize(text: str) -> list[str]:
    return [m.group().casefold() for m in TOKEN_RE.finditer(text)]


def _flush(conn: sqlite3.Connection, counts: Counter) -> None:
    if not counts:
        return
    with conn:
        conn.executemany(
            """INSERT INTO df(term, doc_freq) VALUES (?, ?)
               ON CONFLICT(term) DO UPDATE
               SET doc_freq = doc_freq + excluded.doc_freq""",
            counts.items(),
        )
    counts.clear()


def build_from_texts(texts, db_path: str, flush_every: int = 10000) -> int:
    """Count documents containing each token (not total occurrences)."""
    path = Path(db_path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Choose another --db or delete the old one "
            "explicitly to avoid silently mixing different corpora."
        )
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE df(term TEXT PRIMARY KEY, doc_freq INTEGER NOT NULL)")
        conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
        batch = Counter()
        n_docs = 0
        for text in texts:
            n_docs += 1
            batch.update(set(tokenize(text)))
            if n_docs % flush_every == 0:
                _flush(conn, batch)
                print(f"Processed {n_docs:,} chunks", flush=True)
        _flush(conn, batch)
        with conn:
            conn.execute("INSERT INTO meta(key, value) VALUES ('num_docs', ?)", (n_docs,))
        return n_docs
    finally:
        conn.close()


def build_kilt_idf(db_path: str, max_chunks: int = 100000):
    """max_chunks=0 means scan all ~21m chunks; otherwise use a sample."""
    from datasets import load_dataset

    dataset = load_dataset("dmrau/kilt-128", split="train", streaming=True)
    if max_chunks:
        # Approximate sample; shard/buffer shuffling is not a uniform global sample.
        dataset = dataset.shuffle(seed=42, buffer_size=10000)
        dataset = dataset.take(max_chunks)
    n_docs = build_from_texts((row["content"] for row in dataset), db_path)
    print(f"Saved IDF document frequencies for {n_docs:,} chunks to {db_path}")


def score_text(text: str, db_path: str = "kilt_idf.sqlite") -> list[dict]:
    conn = sqlite3.connect(db_path)
    try:
        result = conn.execute(
            "SELECT value FROM meta WHERE key='num_docs'"
        ).fetchone()
        if result is None:
            raise ValueError("IDF index is incomplete: missing num_docs metadata")
        n_docs = result[0]
        tf = Counter(tokenize(text))
        if not tf:
            return []
        # SQLite default variable limits comfortably accommodate a 128-token chunk.
        placeholders = ",".join("?" for _ in tf)
        known = dict(conn.execute(
            f"SELECT term, doc_freq FROM df WHERE term IN ({placeholders})",
            list(tf),
        ).fetchall())
        results = []
        for term, frequency in tf.items():
            doc_freq = known.get(term, 0)
            idf = math.log((n_docs + 1) / (doc_freq + 1)) + 1
            results.append({
                "term": term,
                "tf": frequency,
                "df": doc_freq,
                "idf": idf,
                "tfidf": frequency * idf,
                "unseen_in_index": term not in known,
            })
        return sorted(results, key=lambda r: r["tfidf"], reverse=True)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--db", default="kilt_idf.sqlite")
    builder.add_argument("--max-chunks", type=int, default=100000,
                         help="0 means full corpus; default: 100000 chunks")
    scorer = subparsers.add_parser("score")
    scorer.add_argument("text")
    scorer.add_argument("--db", default="kilt_idf.sqlite")
    args = parser.parse_args()
    if args.command == "build":
        build_kilt_idf(args.db, args.max_chunks)
    else:
        for row in score_text(args.text, args.db):
            print(f"{row['term']:24} TF={row['tf']:<3} DF={row['df']:<8} "
                  f"IDF={row['idf']:.3f} TF-IDF={row['tfidf']:.3f}" +
                  (" (unseen in index)" if row['unseen_in_index'] else ""))


if __name__ == "__main__":
    main()
