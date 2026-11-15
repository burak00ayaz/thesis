from tf_idf.process_kilt import score_text

context = (
    "Eisenhower followed directive JCS 1067, "
    "but softened the policy by bringing in "
    "400,000 tons of food for civilians."
)

scores = score_text(
    context,
    db_path="tf_idf/kilt.sqlite"
)

for item in scores:
    print(
        item["term"],
        "IDF:", round(item["idf"], 3),
        "TF-IDF:", round(item["tfidf"], 3)
    )