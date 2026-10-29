import csv
from pathlib import Path
import hashlib
import random

from corpus.types import Triplet, Dataset

DATA_FILE = Path(__file__).parent / 'Natural-Questions-Filtered.csv'

def get_triplets():
    with open(DATA_FILE, 'r', encoding='utf-8') as csvfile:
        rows = list(csv.DictReader(csvfile))

    rng = random.Random(42)
    rng.shuffle(rows)
    
    for i, row in enumerate(rows):
        question_id = hashlib.sha256(row["question"].encode('utf-8')).hexdigest()[:8]  # Generate a unique ID based on the question text
        yield Triplet(
            dataset=Dataset.NQ,
            question_id=question_id,
            context=[row['long_answers']],
            question=row['question'],
            answer=row['short_answers'] 
        )

