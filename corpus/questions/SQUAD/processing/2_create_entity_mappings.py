from questions.SQUAD.processing.entity_mapping_prompt import ENTITY_MAPPING_PROMPT
from open_router.api import send_request
from typing import Literal
import json
from pathlib import Path

MODEL = "openai/gpt-5.6-luna"
TEMPERATURE = 0.0
DATA_DIR = Path(__file__).parent.parent / 'data'
DATASET_SPLIT: Literal["train", "validation"] = "train"
DATASET_FILE = DATA_DIR / DATASET_SPLIT / 'context_groups.json'
ENTITY_MAPPINGS_FILE = DATA_DIR / DATASET_SPLIT / 'entity_mappings.json'
STOP_AFTER = 25

with open(DATASET_FILE, 'r') as f:
    data = json.load(f)

try:
    with open(ENTITY_MAPPINGS_FILE, 'r') as f:
        entity_mappings = json.load(f)
except FileNotFoundError:
    entity_mappings = {}

generated = 0

for i, (context_hash, context_dict) in enumerate(data.items()):
    if generated >= STOP_AFTER:
        print(f"Stopping after generating {STOP_AFTER} entity mappings.")
        break
    if entity_mappings.get(context_hash):
        print(f"Skipping example {i} as it already has an entity mapping.")
        continue

    query = ENTITY_MAPPING_PROMPT.replace(
        "{{INPUT_JSON}}", json.dumps(context_dict, ensure_ascii=False)
    )

    response = send_request(query, model=MODEL, temperature=TEMPERATURE)
    entity_mappings[context_hash] = json.loads(response)
    print(f"Response for example {i}:\n{response}\n")
    print(80 * "-")
    generated += 1

with open(ENTITY_MAPPINGS_FILE, 'w', encoding='utf-8') as f:
    json.dump(entity_mappings, f, indent=4, ensure_ascii=False)

print(f"Generated {generated} entity mappings and saved to {ENTITY_MAPPINGS_FILE}.")