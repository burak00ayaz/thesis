import spacy

nlp = spacy.load("en_core_web_sm")

examples = {
    "person_real": [
        "Barack Obama was born in Hawaii.",
        "Angela Merkel served as Chancellor of Germany.",
        "Elon Musk founded several technology companies.",
    ],

    "person_imaginary": [
        "Zorvan Keldrix presented the results at the conference.",
        "Elara Voss joined the research team in Zurich.",
        "Professor Kaelen Morvath developed the new algorithm.",
    ],

    "company_real": [
        "Microsoft released a new version of its software.",
        "Apple acquired the startup for one billion dollars.",
        "Siemens operates in several industrial sectors.",
        "OpenAI released a new language model.",
    ],

    "company_imaginary": [
        "Veltrix Dynamics announced a new processor yesterday.",
        "QuantumForge Technologies opened an office in Munich.",
        "Nexora Biolabs raised 40 million dollars in funding.",
    ],

    "numbers": [
        "The experiment used 128 samples.",
        "The model achieved an accuracy of 87.43 percent.",
        "The company employs approximately 12,500 people.",
        "The temperature increased by 3.7 degrees.",
        "The system contains 32 processing cores.",
    ],

    "dates": [
        "The experiment started on October 14, 2025.",
        "The paper was published in March 2024.",
        "The meeting is scheduled for 17 January 2027.",
        "The project ran from 2021 to 2025.",
        "The server failed at 3:42 PM on Monday.",
    ],

    "identifiers": [
        "The experiment has identifier EXP-2025-00417.",
        "The device serial number is XJ9-42K-771A.",
        "The patient record uses ID PAT-938271.",
        "The commit hash is a8f7c2d91b3e.",
        "The DOI of the paper is 10.1038/s41586-024-07315-1.",
        "The IPv4 address of the server is 192.168.10.42.",
        "The MAC address is 00:1A:2B:3C:4D:5E.",
        "The model version is Qwen2.5-Coder-32B-Instruct.",
    ],

    "medication_real": [
        "The patient was prescribed ibuprofen for the pain.",
        "The doctor prescribed metformin to the patient.",
        "Treatment with amoxicillin was started immediately.",
        "The patient received 20 mg of omeprazole.",
        "Pembrolizumab was administered every three weeks.",
    ],

    "medication_imaginary": [
        "The patient was treated with Velunexor for six weeks.",
        "Doctors administered Trizomab to the patient.",
        "A new drug called Nexaforin showed promising results.",
    ],

    "chemical_real": [
        "The solution contains sodium chloride and water.",
        "Benzene was detected in the sample.",
        "The reaction produces carbon dioxide.",
        "Acetylsalicylic acid was dissolved in ethanol.",
        "The sample contained 4.2 mg of potassium nitrate.",
    ],

    "chemical_formula": [
        "The reaction converts CO2 into CH4.",
        "The solution contains NaCl at a concentration of 0.5 M.",
        "H2SO4 was added slowly to the mixture.",
        "The compound C8H10N4O2 was detected in the sample.",
    ],

    "machine_learning": [
        "The model uses a Transformer architecture with multi-head attention.",
        "We fine-tuned Qwen2.5-Coder-32B-Instruct on HumanEval.",
        "Retrieval-Augmented Generation improves access to external knowledge.",
        "Low-Rank Adaptation reduces the number of trainable parameters.",
    ],

    "computer_architecture": [
        "The processor implements a RISC-V instruction set.",
        "The kernel uses DMA to transfer data into SRAM.",
        "The design contains a vector load-store unit.",
        "The program was compiled using LLVM.",
    ],

    "biology": [
        "The TP53 gene regulates cell division.",
        "CRISPR-Cas9 was used to modify the genome.",
        "Escherichia coli was grown overnight.",
        "The SARS-CoV-2 virus caused the infection.",
    ],
}


output_file = "spacy_noun_chunk_results.txt"

with open(output_file, "w", encoding="utf-8") as f:

    for category, sentences in examples.items():
        f.write("\n" + "=" * 100 + "\n")
        f.write(category.upper() + "\n")
        f.write("=" * 100 + "\n")

        for text in sentences:
            doc = nlp(text)

            f.write(f"\nTEXT: {text}\n")

            f.write("\nTOKENS:\n")
            for token in doc:
                f.write(
                    f"{token.text:<30}"
                    f"lemma={token.lemma_:<20}"
                    f"pos={token.pos_:<8}"
                    f"tag={token.tag_:<8}"
                    f"dep={token.dep_:<12}"
                    f"shape={token.shape_:<15}"
                    f"alpha={str(token.is_alpha):<6}"
                    f"stop={str(token.is_stop):<6}"
                    "\n"
                )

            f.write("\nNOUN CHUNKS:\n")

            noun_chunks = list(doc.noun_chunks)

            if noun_chunks:
                for chunk in noun_chunks:
                    f.write(
                        f"  {chunk.text:<45}"
                        f"root={chunk.root.text:<20}"
                        f"root_dep={chunk.root.dep_:<15}"
                        f"root_head={chunk.root.head.text}\n"
                    )
            else:
                f.write("  <none>\n")

            f.write("-" * 100 + "\n")

print(f"Results written to {output_file}")