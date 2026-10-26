from questions.types import Dataset, Triplet

def triplets(dataset: Dataset):
    if dataset == Dataset.SQUAD:
        from questions.SQUAD.squad import get_triplets
        return get_triplets(entity_mapping=False)
    elif dataset == Dataset.SQUAD_MAPPED:
        from questions.SQUAD.squad import get_triplets
        return get_triplets(entity_mapping=True)
    elif dataset == Dataset.SYNTHWORLDS_SM:
        from questions.SynthWorlds.synthworlds import get_triplets
        return get_triplets(dataset)
    elif dataset == Dataset.SYNTHWORLDS_RM:
        from questions.SynthWorlds.synthworlds import get_triplets
        return get_triplets(dataset)
    elif dataset == Dataset.NQ:
        from questions.NQ.nq import get_triplets
        return get_triplets()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


if __name__ == "__main__":
    # squad_triplets = triplets(Dataset.SQUAD)
    # squad_mapped_triplets = triplets(Dataset.SQUAD_MAPPED)

    # for i, (triplet_real, triplet_mapped) in enumerate(zip(squad_triplets, squad_mapped_triplets)):
    #     print(f"Real Triplet {i}:")
    #     print(f"Question ID: {triplet_real.question_id}")
    #     print(f"Context: {triplet_real.context}")
    #     print(f"Question: {triplet_real.question}")
    #     print(f"Answer: {triplet_real.answer}")
    #     print(80 * "-")
    #     print(f"Mapped Triplet {i}:")
    #     print(f"Question ID: {triplet_mapped.question_id}")
    #     print(f"Context: {triplet_mapped.context}")
    #     print(f"Question: {triplet_mapped.question}")
    #     print(f"Answer: {triplet_mapped.answer}")
    #     print(80 * "=")
    #     if i == 122:
    #         break

    nq_triplets = triplets(Dataset.NQ)

    for i, triplet in enumerate(nq_triplets):
        print(f"Triplet {i}:")
        print(f"Question ID: {triplet.question_id}")
        print(f"Context: {triplet.context}")
        print(f"Question: {triplet.question}")
        print(f"Answer: {triplet.answer}")
        print(80 * "=")
        if i == 9:
            break