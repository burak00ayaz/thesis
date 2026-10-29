from corpus.types import Dataset, Triplet
from corpus.filtering.util import filtering_wrapper

def triplets(dataset: Dataset, entity_questions=False):
    if dataset == Dataset.SQUAD:
        from corpus.questions.SQUAD.squad import get_triplets
        generator = get_triplets(entity_mapping=False)
    elif dataset == Dataset.SQUAD_MAPPED:
        from corpus.questions.SQUAD.squad import get_triplets
        generator = get_triplets(entity_mapping=True)
    elif dataset == Dataset.SYNTHWORLDS_SM:
        from corpus.questions.SynthWorlds.synthworlds import get_triplets
        generator = get_triplets(dataset)
    elif dataset == Dataset.SYNTHWORLDS_RM:
        from corpus.questions.SynthWorlds.synthworlds import get_triplets
        generator = get_triplets(dataset)
    elif dataset == Dataset.NQ:
        from corpus.questions.NQ.nq import get_triplets
        generator = get_triplets()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    generator = filtering_wrapper(generator, dataset, filter_entity_questions=entity_questions)
    return generator

if __name__ == "__main__":
    nq_triplets = triplets(Dataset.NQ, entity_questions=True)

    for i, (triplet, entity_label) in enumerate(nq_triplets):
        print(f"Triplet {i}:")
        print(f"Question ID: {triplet.question_id}")
        print(f"Context: {triplet.context}")
        print(f"Question: {triplet.question}")
        print(f"Answer: {triplet.answer}")
        print(f"Detected Entity Label: {entity_label}")
        print(80 * "=")
        if i == 10:
            break