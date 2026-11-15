from corpus.questions.questions import triplets
from corpus.types import Dataset, Triplet
from numerical_detector import extract_numerical_information, NumericalMention


squad_ds = triplets(Dataset.SQUAD, entity_questions=True)

count = 0

for i, (triplet, entity_label) in enumerate(squad_ds):
    print(f"Triplet {i+1}:")
    if i >= 220:
        break
    if entity_label not in ["TIME", "NUMERIC"]:
        continue


    print(f"Question: {triplet.question}")
    print(f"Context: {triplet.context}")
    print(f"Answer: {triplet.answer}")
    print(f"Entity Label: {entity_label}")

    example_context = " ".join(triplet.context)
    numerical_mentions = extract_numerical_information(example_context)
    for mention in numerical_mentions:
        print(mention.to_dict())
    side_channel = "|".join([mention.text for mention in numerical_mentions])
    print(f"Side Channel: {side_channel}")
    print("\n" + "-" * 80 + "\n")
    count += 1


print(f"Processed {count} triplets with entity labels 'TIME' or 'NUMERIC'.")