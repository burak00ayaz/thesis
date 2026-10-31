from pathlib import Path
import json
import time

from corpus.entity.falcon_linker import FalconEntityLinker


linker = FalconEntityLinker()


# read a json file that is located under parent/filtering/entity_questions
with open(Path(__file__).parent.parent / "filtering" / "entity_questions" / "squad.json", "r") as f:
    entity_questions = json.load(f)

for question_id, entity_question in entity_questions.items():
    if entity_question['label'] not in ['PERSON']:
        continue

    print(f"Question ID: {question_id}")
    print(f"Context: {' '.join(entity_question['context'])}")
    print(f"Question: {entity_question['question']}")
    print(f"Answer Entity: {entity_question['answer_entity']}")
    print(f"Label: {entity_question['label']}")

    search_text = " ".join(entity_question["context"])
    print(80 * "-")
    print(f"Search text: {search_text}")
    all_results = linker.link(text=search_text)

    # print with indentation for better readability
    # print(f"All detected entities: {json.dumps([entity.__dict__ for entity in all_results], indent=4)}")
    
    # match linked entity name with answer entity name
    answer_entity = [
        entity for entity in all_results if entity_question['answer_entity'].lower() in entity.name.lower() 
    ]
    
    print(f"Answer entity detected: {answer_entity}")
    time.sleep(2)

    print(80 * "=")
    print("\n")
