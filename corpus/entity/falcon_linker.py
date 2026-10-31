
import re
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class LinkedEntity:
    name: str
    wikidata_id: str
    wikidata_uri: str


class FalconEntityLinker:
    API_URL = "https://labs.tib.eu/falcon/falcon2/api"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )

        self.session.mount(
            "https://",
            HTTPAdapter(max_retries=retry),
        )

    def request(self, text: str) -> dict[str, Any]:
        """Send text to Falcon and return the raw response."""
        response = self.session.post(
            self.API_URL,
            params={"mode": "long"},
            json={"text": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


    def link(self, text: str) -> list[LinkedEntity]:
        """Extract Wikidata-linked entities from text."""
        data = self.request(text)
        entities = []

        for entry in data.get("entities_wikidata", []):
            uri = entry["URI"].strip("<>")
            name = entry["surface form"]

            match = re.search(r"/(Q\d+)$", uri)
            if not match:
                continue

            entities.append(
                LinkedEntity(
                    name=name,
                    wikidata_id=match.group(1),
                    wikidata_uri=uri,
                )
            )

        return entities


    def close(self):
        self.session.close()


if __name__ == "__main__":
    linker = FalconEntityLinker()

    text = "Apple was founded by Steve Jobs."

    try:
        for entity in linker.link(text):
            print(entity)
    finally:
        linker.close()
