from urllib.parse import quote
import requests
import time
from nq_google_stuff.nq_triplets import nq_original_triplets
import json


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
        start: str = "20230901",
        end: str = "20230930",
    ) -> dict:

        popularity = self.get_pageviews(wikipedia_title, start, end)

        return {
            "wikipedia_title": wikipedia_title,
            "popularity": popularity,
        }


if __name__ == "__main__":
    service = EntityPopularityService(
        contact_email="burakayax@gmail.com"
    )

    popularity_results = {}

    for i, (triplet, wikipedia_title) in enumerate(nq_original_triplets()):
        print(f"Processing triplet {i}...")
        result = service.get_popularity(wikipedia_title=wikipedia_title)
        popularity_results[triplet.question_id] = result
        print(result)
        time.sleep(1)
        if i == 201:
            break

    # Save the results to a JSON file with indentation for better readability
    with open("popularity_results.json", "w", encoding="utf-8") as f:
        json.dump(popularity_results, f, ensure_ascii=False, indent=4)