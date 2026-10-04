from transformers import AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
OUTPUT_FILE = "fragmentation_results.txt"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def fragmentation(text: str):
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    num_tokens = len(tokens)
    num_chars = len(text)
    num_words = len(text.split())

    return {
        "text": text,
        "tokens": tokens,
        "num_tokens": num_tokens,
        "num_chars": num_chars,
        "num_words": num_words,
        "tokens_per_word": num_tokens / max(num_words, 1),
        "tokens_per_char": num_tokens / max(num_chars, 1),
    }


examples = [
    # ------------------------------------------------------------------
    # PERSON NAMES - REAL
    # ------------------------------------------------------------------
    "Barack Obama",
    "Angela Merkel",
    "Elon Musk",

    # ------------------------------------------------------------------
    # PERSON NAMES - IMAGINARY
    # ------------------------------------------------------------------
    "Zorvan Keldrix",
    "Elara Voss",
    "Kaelen Morvath",

    # ------------------------------------------------------------------
    # COMPANIES / ORGANIZATIONS - REAL
    # ------------------------------------------------------------------
    "Microsoft",
    "Apple",
    "Siemens",
    "OpenAI",

    # ------------------------------------------------------------------
    # COMPANIES / ORGANIZATIONS - IMAGINARY
    # ------------------------------------------------------------------
    "Veltrix Dynamics",
    "QuantumForge Technologies",
    "Nexora Biolabs",

    # ------------------------------------------------------------------
    # NUMBERS
    # ------------------------------------------------------------------
    "128",
    "87.43 percent",
    "12,500",
    "3.7 degrees",
    "32",

    # ------------------------------------------------------------------
    # DATES / TIMES
    # ------------------------------------------------------------------
    "October 14, 2025",
    "March 2024",
    "17 January 2027",
    "2021 to 2025",
    "3:42 PM",
    "Monday",

    # ------------------------------------------------------------------
    # IDENTIFIERS
    # ------------------------------------------------------------------
    "EXP-2025-00417",
    "XJ9-42K-771A",
    "PAT-938271",
    "a8f7c2d91b3e",
    "10.1038/s41586-024-07315-1",
    "192.168.10.42",
    "00:1A:2B:3C:4D:5E",
    "Qwen2.5-Coder-32B-Instruct",

    # ------------------------------------------------------------------
    # MEDICATIONS - REAL
    # ------------------------------------------------------------------
    "ibuprofen",
    "metformin",
    "amoxicillin",
    "omeprazole",
    "Pembrolizumab",

    # ------------------------------------------------------------------
    # MEDICATIONS - IMAGINARY
    # ------------------------------------------------------------------
    "Velunexor",
    "Trizomab",
    "Nexaforin",

    # ------------------------------------------------------------------
    # CHEMICAL SUBSTANCES
    # ------------------------------------------------------------------
    "sodium chloride",
    "water",
    "benzene",
    "carbon dioxide",
    "acetylsalicylic acid",
    "ethanol",
    "potassium nitrate",

    # ------------------------------------------------------------------
    # CHEMICAL FORMULAS
    # ------------------------------------------------------------------
    "CO2",
    "CH4",
    "NaCl",
    "H2SO4",
    "C8H10N4O2",

    # ------------------------------------------------------------------
    # MACHINE LEARNING
    # ------------------------------------------------------------------
    "Transformer",
    "multi-head attention",
    "Qwen2.5-Coder-32B-Instruct",
    "HumanEval",
    "Retrieval-Augmented Generation",
    "Low-Rank Adaptation",

    # ------------------------------------------------------------------
    # COMPUTER ARCHITECTURE
    # ------------------------------------------------------------------
    "RISC-V",
    "DMA",
    "SRAM",
    "vector load-store unit",
    "LLVM",

    # ------------------------------------------------------------------
    # BIOLOGY
    # ------------------------------------------------------------------
    "TP53",
    "CRISPR-Cas9",
    "Escherichia coli",
    "SARS-CoV-2",
]


results = [fragmentation(x) for x in examples]

# Sort from most fragmented to least fragmented.
# Change to "tokens_per_word" if that is the metric you want to analyze.
results.sort(
    key=lambda x: x["tokens_per_word"],
    reverse=True,
)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Number of examples: {len(results)}\n")
    f.write("=" * 100 + "\n\n")

    for result in results:
        f.write(f"Text:            {result['text']}\n")
        f.write(f"Tokens:          {result['tokens']}\n")
        f.write(f"Num tokens:      {result['num_tokens']}\n")
        f.write(f"Num chars:       {result['num_chars']}\n")
        f.write(f"Num words:       {result['num_words']}\n")
        f.write(f"Tokens / word:   {result['tokens_per_word']:.4f}\n")
        f.write(f"Tokens / char:   {result['tokens_per_char']:.4f}\n")
        f.write("-" * 100 + "\n")


print(f"Results written to {OUTPUT_FILE}")