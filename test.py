import json

with open("evaluation/wikilinker/linked_entities.json", "r", encoding="utf-8") as f:
    linked_entities = json.load(f)

with open("corpus/filtering/entity_questions/squad_validation.json", "r", encoding="utf-8") as f:
    entity_questions = json.load(f)
    # filter out TIME and DATE labels
    entity_questions = {qid: q for qid, q in entity_questions.items() if q["label"] not in ["TIME", "NUMERIC"]}


# get the intersection of question IDs in both files
common_question_ids = set(linked_entities.keys()).intersection(set(entity_questions.keys()))

print(f"Number of question IDs in linked_entities.json: {len(linked_entities)}")
print(f"Number of question IDs in entity_questions.json: {len(entity_questions)}")
print(f"Number of common question IDs: {len(common_question_ids)}")


# create a new dictionary with only the common question IDs
filtered_linked_entities = {qid: linked_entities[qid] for qid in common_question_ids}

# add the label and answer_entity from entity_questions to filtered_linked_entities
for qid in filtered_linked_entities:
    filtered_linked_entities[qid]["label"] = entity_questions[qid]["label"]

with open("evaluation/wikilinker/filtered_linked_entities.json", "w", encoding="utf-8") as f:
    json.dump(filtered_linked_entities, f, ensure_ascii=False, indent=4)
    print(f"Saved filtered linked entities with {len(filtered_linked_entities)} question IDs to filtered_linked_entities.json")