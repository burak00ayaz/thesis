
import requests
import time
import json
import os
from datasets import load_dataset

USER_AGENT = "ThesisQAResearch/0.1 (contact: burakayax@gmail.com)"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def get_wikipedia_context(qid: str) -> dict | None:
    time.sleep(1)  # Sleep for 1 second to respect API rate limits

    # Step 1: Resolve Wikidata QID to Wikipedia article
    response = session.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json",
        },
        timeout=20,
    )
    response.raise_for_status()

    try: 
        entity = response.json()["entities"].get(qid, {})
        title = entity.get("sitelinks", {}).get(
            "enwiki", {}
        ).get("title")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error resolving QID {qid}: {e}")
        return None

    print(f"Resolved QID {qid} to Wikipedia title: {title}")
    time.sleep(1)  # Sleep for 1 second to respect API rate limits

    if not title:
        return None

    # Step 2: Retrieve the Wikipedia article text
    response = session.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": 1,
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        },
        timeout=20,
    )
    response.raise_for_status()

    pages = response.json()["query"]["pages"]
    page = pages[0]

    if "missing" in page or not page.get("extract"):
        return None

    return {
        "qid": qid,
        "title": page["title"],
        "context": page["extract"],
        "pageid": page["pageid"],
        "source": "wikipedia",
    }


def fetch_and_save_wikipedia_articles(stop_after: int, output_json_file: str):
    ds = load_dataset("google/granola-entity-questions", token=True)["train"]
    ds = ds.shuffle(seed=42)  # Shuffle the dataset for randomness

    # get current file directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_json_file_path = os.path.join(current_dir, output_json_file)
    output_data_dir = os.path.join(current_dir, "data")

    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    output_json = {}
    count = 0

    for i, example in enumerate(ds):
        if count >= stop_after:
            break
        question_entity = example["question_entity"]
        question_entity_qid = example["question_entity_qid"]


        # check if the article for this QID already exists
        article_path = os.path.join(output_data_dir, f"{question_entity_qid}.txt")
        if os.path.exists(article_path):
            print(f"INFO: Article for {question_entity} {question_entity_qid} already exists. Skipping.")
            count += 1
            output_json[question_entity_qid] = {
                "count": count,
                "question_entity": question_entity,
                "article_path": article_path,
            }
            continue
        else:
            wikipedia_context = get_wikipedia_context(question_entity_qid)

        print(f"{i+1}: Wikipedia Context for {question_entity} {question_entity_qid}:")
        if wikipedia_context is not None:
            answer = example["answer"]
            if answer.lower() in wikipedia_context['context'].lower():
                print("INFO: The answer is present in the Wikipedia context.")
                article_path = os.path.join(output_data_dir, f"{question_entity_qid}.txt")

                with open(article_path, "w") as f:
                    f.write(wikipedia_context['context'])

                count += 1
                output_json[question_entity_qid] = {
                    "count": count,
                    "question_entity": question_entity,
                    "article_path": article_path,
                }
                
            else:
                print("INFO: The answer is NOT present in the Wikipedia context.")
        else:
            print("WARN: No context found.")

    with open(output_json_file_path, "w") as f:
        json.dump(output_json, f, indent=4)
    print(f"INFO: Saved output JSON to {output_json_file_path}")


if __name__ == "__main__":
    STOP_AFTER = 200  # Set the number of articles to fetch
    OUTPUT_JSON_FILE = "wikipedia_articles.json"  # Set the output JSON file name
    fetch_and_save_wikipedia_articles(STOP_AFTER, OUTPUT_JSON_FILE)