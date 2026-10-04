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

    synthworlds_sm_triplets = triplets(Dataset.SYNTHWORLDS_SM)
    synthworlds_rm_triplets = triplets(Dataset.SYNTHWORLDS_RM)

    for i, (triplet_sm, triplet_rm) in enumerate(zip(synthworlds_sm_triplets, synthworlds_rm_triplets)):
        print(f"SynthWorlds SM Triplet {i}:")
        print(triplet_sm)
        print(80 * "-")
        print(f"SynthWorlds RM Triplet {i}:")
        print(triplet_rm)
        print(80 * "=")
        if i == 9:
            break