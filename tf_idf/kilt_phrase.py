"""Compute chunk-level document frequency for a fixed list of multiword phrases.

Install: pip install datasets pyahocorasick
Usage: python kilt_phrase_idf.py --phrases entities.txt --db kilt_phrases.sqlite
       python kilt_phrase_idf.py --phrases entities.txt --db test.sqlite --max-chunks 10000

Each nonempty line in entities.txt contains one phrase.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import re
import sqlite3

import ahocorasick

TOKEN_RE = re.compile(
    r"(?u)\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\w+(?:[-'’]\w+)*\b"
)


def canonical(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def create_matcher(phrases: list[str]):
    # Add leading/trailing spaces to enforce whole-token matches.
    a = ahocorasick.Automaton()
    seen = set()
    for phrase in phrases:
        key = canonical(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        a.add_word(f" {key} ", key)
    if not seen:
        raise ValueError("No valid phrases supplied")
    a.make_automaton()
    return a, sorted(seen)


def scan(text: str, automaton) -> set[str]:
    normalized = f" {canonical(text)} "
    return {key for _, key in automaton.iter(normalized)}


def build(phrases_path: str, db_path: str, max_chunks: int = 0):
    from datasets import load_dataset

    phrases = Path(phrases_path).read_text(encoding="utf-8").splitlines()
    automaton, phrases = create_matcher(phrases)

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS phrase_df (phrase TEXT PRIMARY KEY, df INTEGER NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        if conn.execute("SELECT COUNT(*) FROM meta WHERE key='num_docs'").fetchone()[0]:
            raise RuntimeError("This DB has already been counted. Use a new path to avoid double-counting.")
        conn.executemany(
            "INSERT OR IGNORE INTO phrase_df (phrase, df) VALUES (?, 0)",
            [(p,) for p in phrases],
        )
        conn.commit()
        ds = load_dataset("dmrau/kilt-128", split="train", streaming=True)
        counter = Counter()
        processed = 0
        def flush():
            # One transaction: counts and N are always in sync after interruption.
            with conn:
                conn.executemany(
                    "UPDATE phrase_df SET df = df + ? WHERE phrase = ?",
                    [(freq, phrase) for phrase, freq in counter.items()],
                )
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES ('num_docs', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(processed),),
                )
            counter.clear()
        try:
            for row in ds:
                if max_chunks and processed >= max_chunks:
                    break
                counter.update(scan(row["content"], automaton))
                processed += 1
                if processed % 10000 == 0:
                    flush()
                    print(f"Processed {processed:,} chunks", flush=True)
        except KeyboardInterrupt:
            print("Interrupted: saving completed and current counts...", flush=True)
        finally:
            flush()
        print(f"Saved {processed:,} chunk frequencies in {db_path}")


def show(db_path: str, phrase: str):
    key = canonical(phrase)
    with sqlite3.connect(db_path) as conn:
        res = conn.execute("SELECT df FROM phrase_df WHERE phrase = ?", (key,)).fetchone()
        num_row = conn.execute("SELECT value FROM meta WHERE key = 'num_docs'").fetchone()
    if num_row is None:
        raise RuntimeError("No completed corpus scan is recorded")
    if res is None:
        raise KeyError(f"Phrase {phrase!r} was not included in candidates list")
    n, df = int(num_row[0]), res[0]
    idf = math.log((1 + n) / (1 + df)) + 1
    #print(f"Phrase: {phrase}\nChunks: {n:,}\nDocument frequency: {df:,}\nIDF: {idf:.4f}")
    return idf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases", help="UTF-8 file, one phrase per line")
    parser.add_argument("--db", required=True)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--lookup", help="Query an indexed phrase instead of building")
    args = parser.parse_args()
    if args.lookup:
        show(args.db, args.lookup)
    else:
        if not args.phrases:
            parser.error("--phrases required to build")
        build(args.phrases, args.db, args.max_chunks)
