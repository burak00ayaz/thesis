from questions.SQUAD.processing.entity_mapping_prompt import ENTITY_MAPPING_PROMPT
from open_router.api import send_request

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
DATASET_FILE = 'squad_train.json'

with open(DATA_DIR / DATASET_FILE, 'r') as f:
    data = json.load(f)


print(len(data), "examples loaded from", DATASET_FILE)
exit(0)


for i in range(3):
    context_dict = data[i]

    query = ENTITY_MAPPING_PROMPT.replace(
        "{{INPUT_JSON}}", json.dumps(context_dict, ensure_ascii=False)
    )

    response = send_request(query)
    print(f"Response for example {i}:\n{response}\n")
    print(80 * "-")