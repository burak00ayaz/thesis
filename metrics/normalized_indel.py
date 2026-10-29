from rapidfuzz.fuzz import partial_ratio_alignment


# substring-aligned normalized Indel similarity
def substring_aligned_normalized_indel(
    expected_answer: str,
    model_output: str,
) -> dict:
    alignment = partial_ratio_alignment(
        expected_answer.lower(),
        model_output.lower(),
    )

    matched_substring = model_output[
        alignment.dest_start:alignment.dest_end
    ]

    return {
        "similarity": alignment.score / 100,
        "matched_substring": matched_substring,
    }

if __name__ == "__main__":
    expected = "Barack Obama"

    outputs = [
        "The answer is Barack Obama.",
        "The answer is Barak Obama.",
        "It was Barack Obam.",
        "The answer is Michelle Obama.",
        "I don't know.",
        "The answer is Barack Obamaga."
    ]

    for output in outputs:
        print(output, substring_aligned_normalized_indel(expected, output))


    #example: lexically similar but different entities
    expected = "University of California, Berkeley"
    outputs = [
        "University of California, Los Angeles",
        "University of Berkeley, California"
    ]

    for output in outputs:
        print(output, substring_aligned_normalized_indel(expected, output))
