
from urllib.parse import quote
import json
import requests
import time
from corpus.questions.SQUAD.squad_validation import get_by_question_id


class EntityPopularityService:
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    PAGEVIEWS_API = (
        "https://wikimedia.org/api/rest_v1/metrics/"
        "pageviews/per-article"
    )

    def __init__(self, contact_email: str):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                f"ThesisEntityPopularity/0.1 "
                f"(contact: {contact_email})"
            )
        })

    def get_pageviews(
        self,
        title: str,
        start: str = "20230901",
        end: str = "20230930",
    ) -> int | None:
        encoded_title = quote(
            title.replace(" ", "_"),
            safe="",
        )

        url = (
            f"{self.PAGEVIEWS_API}/"
            f"en.wikipedia.org/all-access/user/"
            f"{encoded_title}/daily/{start}/{end}"
        )

        response = self.session.get(url, timeout=30)

        if response.status_code == 404:
            return None

        response.raise_for_status()

        items = response.json().get("items", [])

        return sum(item["views"] for item in items)

    def get_popularity(
        self,
        wikipedia_title: str | None = None,
        start: str = "20160101",
        end: str = "20161231",
    ) -> int | None:

        popularity = self.get_pageviews(wikipedia_title, start, end)
        return popularity


if __name__ == "__main__":
    service = EntityPopularityService(
        contact_email="burakayax@gmail.com"
    )

    entity_questions_path = "corpus/filtering/entity_questions/squad_validation.json"
    output_path = "evaluation/popularity/entity_popularity.json"

    with open(entity_questions_path, "r", encoding="utf-8") as f:
        entity_questions = json.load(f)
        print(f"Loaded {len(entity_questions)} entity questions from {entity_questions_path}")

    entity_popularity = {}
    entity_popularity_detailed = {}

    for i, (question_id, entity_question) in enumerate(entity_questions.items()):
        question = get_by_question_id(question_id)
        title = question["title"]

        if entity_question["label"] != "PERSON":
            continue  # Skip non-PERSON entities for now

        popularity = service.get_popularity(wikipedia_title=title)
        entity_popularity[question_id] = popularity
        entity_popularity_detailed[question_id] = {
            "title": title,
            "popularity": popularity,
            "answer_entity": entity_question["answer_entity"],
            "label": entity_question["label"],
        }

        print(f"Question ID: {question_id}, Title: {title}, Popularity: {entity_popularity[question_id]}")
        time.sleep(1)  # Sleep for 1 second to avoid hitting API rate limits

        if i >= 10:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entity_popularity, f, ensure_ascii=False, indent=4)
        print(f"Saved entity popularity data to {output_path}")

    with open("evaluation/popularity/entity_popularity_detailed.json", "w", encoding="utf-8") as f:
        json.dump(entity_popularity_detailed, f, ensure_ascii=False, indent=4)
        print(f"Saved detailed entity popularity data to evaluation/popularity/entity_popularity_detailed.json")
