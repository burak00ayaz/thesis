import math
import requests
import time
import json

def get_inlink_popularity(title: str, session: requests.Session):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "backlinks",
        "bltitle": title,
        "blnamespace": 0,
        "blfilterredir": "nonredirects",
        "bllimit": "max",
        "format": "json",
    }

    count = 0

    while True:
        time.sleep(0.5)  # To respect API rate limits
        response = session.get(
            url,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        count += len(data["query"]["backlinks"])

        if "continue" not in data:
            break

        params.update(data["continue"])

    return {
        "title": title,
        "incoming_links": count,
        "popularity": math.log1p(count),
    }


if __name__ == "__main__":
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "ThesisInlinkPopularity/0.1 "
            "(contact: burakayax@gmail.com)"
        )
    })

    with open("evaluation/wikilinker/linked_entities.json", "r", encoding="utf-8") as f:
        link_results = json.load(f)

    try:
        with open("evaluation/wikilinker/entity_inlink_popularity.json", "r", encoding="utf-8") as f:
            popularity_results = json.load(f)
            last_processed_question_id = list(popularity_results.keys())[-1]
            print(f"Resuming from last processed question ID: {last_processed_question_id}")
    except FileNotFoundError:
        popularity_results = {}
        last_processed_question_id = None
        print("No previously processed popularity results found. Starting fresh.")


    for i, (question_id, question) in enumerate(link_results.items()):
        if question_id in popularity_results:
            continue  # Skip already processed questions

        wiki_title = question["link"]["wikipedia_url"].split("/")[-1]
        print(f"Processing {i + 1}/{len(link_results)}: {wiki_title} for question ID {question_id}...")
        popularity_info = get_inlink_popularity(wiki_title, session)
        popularity_results[question_id] = {
            "title": popularity_info["title"],
            "url": question["link"]["wikipedia_url"],
            "incoming_links": popularity_info["incoming_links"],
            "popularity": popularity_info["popularity"]
        }
        with open("evaluation/wikilinker/entity_inlink_popularity.json", "w", encoding="utf-8") as f:
            json.dump(popularity_results, f, ensure_ascii=False, indent=4)

    
    with open("evaluation/wikilinker/entity_inlink_popularity.json", "w", encoding="utf-8") as f:
        json.dump(popularity_results, f, ensure_ascii=False, indent=4)
        print(f"Saved inlink popularity results for {len(popularity_results)} entities to evaluation/wikilinker/entity_inlink_popularity.json")