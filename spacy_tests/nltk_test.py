import nltk

nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

text = "Barack Obama is the first president of the United States with an African American background."
text = f"The model Qwen2.5-Coder-32B-Instruct achieved 87.43% on HumanEval."
text = "North Atlantic Treaty Organization, is an intergovernmental military and political alliance between 32 countries from North America and Europe."

words = word_tokenize(text)

print(words)