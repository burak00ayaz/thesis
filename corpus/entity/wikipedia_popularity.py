
import requests
from urllib.parse import quote
import time


class WikipediaPopularity:
    WIKIDATA_URL = "https://www.wikidata.org/wiki/Special:EntityData"
    PAGEVIEWS_URL = (
        "https://wikimedia.org/api/rest_v1/metrics/"
        "pageviews/per-article"
    )

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ThesisEntityResearch/1.0 (contact: burakayax@gmail.com)",
            "Accept": "application/json",
        })

    def get_wikipedia_title(
        self,
        wikidata_id: str,
        language: str = "en",
    ) -> str | None:
        """Resolve a Wikidata QID to a Wikipedia article title."""
        response = self.session.get(
            f"{self.WIKIDATA_URL}/{wikidata_id}.json",
            timeout=30,
        )
        response.raise_for_status()

        entity = response.json()["entities"][wikidata_id]
        sitelink = entity.get("sitelinks", {}).get(f"{language}wiki")

        return sitelink["title"] if sitelink else None

    def get_pageviews(
        self,
        wikidata_id: str,
        start: str = "20250101",
        end: str = "20251231",
        language: str = "en",
    ) -> int | None:
        """Get total article views for a fixed date range."""
        title = self.get_wikipedia_title(wikidata_id, language)

        if title is None:
            return None

        article = quote(title.replace(" ", "_"), safe="")

        url = (
            f"{self.PAGEVIEWS_URL}/"
            f"{language}.wikipedia.org/"
            f"all-access/user/"
            f"{article}/monthly/{start}/{end}"
        )

        response = self.session.get(url, timeout=30)

        if response.status_code == 404:
            return None

        response.raise_for_status()

        return sum(
            item["views"]
            for item in response.json()["items"]
        )

    def close(self):
        self.session.close()


if __name__ == "__main__":
    popularity = WikipediaPopularity()

    try:
        for qid in ["Q76", "Q42", "Q64"]:
            views = popularity.get_pageviews(qid)
            print(f"{qid}: {views:,}" if views is not None else f"{qid}: N/A")
            time.sleep(2)  # Be polite to the API
    finally:
        popularity.close()
