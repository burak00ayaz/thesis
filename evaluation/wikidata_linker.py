import requests


class WikidataLinker:
    API_URL = "https://www.wikidata.org/w/api.php"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ThesisEntityLinking/0.1 (contact: your-email@example.com)"
        })

    def link(self, entity: str, limit: int = 5) -> list[dict]:
        response = self.session.get(
            self.API_URL,
            params={
                "action": "wbsearchentities",
                "search": entity,
                "language": "en",
                "type": "item",
                "limit": limit,
                "format": "json",
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("search", [])