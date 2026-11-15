from models.model import get_model, Model
from corpus.types import Triplet, Dataset
from corpus.questions.questions import triplets

pisco_model = get_model(Model.PISCO)

squad_ds = triplets(Dataset.SQUAD, entity_questions=False)

reconstruction_prompt = (
    "Reconstruct the original document from the compressed "
    "representation as accurately as possible. "
    "Preserve the exact wording, names, dates, numbers, "
    "and other identifiers. "
    "Do not summarize or paraphrase the document. "
    "Output only the reconstructed text as a single passage."
)

question = "What percentage of people in Egypt support stoning a person who commits adultery?"

test_context = [
    'Following the German unconditional surrender, Eisenhower was appointed Military Governor of the U.S. Occupation Zone, based at the IG Farben Building in Frankfurt am Main. He had no responsibility for the other three zones, controlled by Britain, France and the Soviet Union, except for the city of Berlin, which was managed by the Four-Power Authorities through the Allied Kommandatura as the governing body. Upon discoveryof the Nazi concentration camps, he ordered camera crews to document evidence of the atrocities in them for use in the Nuremberg Trials.', 
    'He reclassified German prisoners of war (POWs) in U.S. custody as Disarmed Enemy Forces (DEFs), who were no longer subject to the Geneva Convention. Eisenhower followed the orders laid down by the Joint Chiefs of Staff (JCS) in directive JCS 1067, but softened them by bringing in 4,000,000 tons of food for civilians and allowing more fraternization.', 
    'In response to the devastation in Germany, including food shortages and an influx of refugees, he arranged distribution of American food and medical equipment. His actions reflected the new American attitudes of the German people as Nazi victims not villains, while aggressively purging the ex-Nazis.'
]

question = "4,000,000 | How many tons of food did Eisenhower bring in?"

####################33


# Only the relevant original chunk
answer_1 = pisco_model.answer_question(
    question,
    [test_context[1]]
)

# All three original chunks
answer_2 = pisco_model.answer_question(
    question,
    test_context
)

print("Relevant chunk only:", answer_1)
print("All chunks:", answer_2)

exit(0)
#####################


pisco_output = pisco_model.answer_question(question, test_context, max_new_tokens=128)
print(f"Question: {question}")
print(f"Context: {test_context}")
print(f"Answer: {pisco_output}")

exit(0)




for i, (triplet, _) in enumerate(squad_ds):
    print(f"Processing Triplet {i}:")
    print(f"Context: {triplet.context}")

    for context_chunk in triplet.context:
        pisco_output = pisco_model.answer_question(reconstruction_prompt, [context_chunk], max_new_tokens=512)
        print(f"Reconstructed Output: {pisco_output}")
    print("" + "=" * 80 + "\n")