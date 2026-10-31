
from urllib.parse import quote

import requests


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

        popularity = self.get_pageviews(title, start, end)

        return {
            "title": title,
            "popularity": popularity,
        }


if __name__ == "__main__":
    service = EntityPopularityService(
        contact_email="burakayax@gmail.com"
    )

    for title in ["German Empire"]:
        result = service.get_popularity(wikipedia_title=title)
        print(result)
