import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse
from corpus.questions.SQUAD.squad_validation import get_by_question_id
from corpus.types import Dataset, Triplet
from corpus.questions.questions import triplets
import time
import json
from evaluation.util import normalize_string

class WikipediaEntityLinker:
    API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, contact_email: str):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                f"ThesisEntityLinker/0.1 "
                f"(contact: {contact_email})"
            )
        })
        self.cache = {}

    @staticmethod
    def normalize(text: str) -> str:
        normalized = " ".join(
            text.replace("_", " ").casefold().split()
        )
        return normalize_string(normalized)  # Further normalize using the utility function

    def get_links(self, title: str) -> list[dict]:
        if title in self.cache:
            return self.cache[title]

        response = self.session.get(
            self.API_URL,
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "format": "json",
                "formatversion": 2,
                "redirects": 1,
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        if "error" in data:
            raise ValueError(data["error"])

        soup = BeautifulSoup(
            data["parse"]["text"], "html.parser"
        )

        links = []

        for a in soup.find_all("a", href=True):
            path = urlparse(a["href"]).path

            if not path.startswith("/wiki/"):
                continue

            target = unquote(path[len("/wiki/"):])

            # Ignore non-article namespaces.
            if ":" in target:
                continue

            links.append({
                "mention": a.get_text(" ", strip=True),
                "target": target,
            })

        self.cache[title] = links
        return links

    def link_entity(
        self,
        source_title: str,
        entity_name: str,
    ) -> dict | None:

        entity = self.normalize(entity_name)
        links = self.get_links(source_title)

        # Prefer anchor-text matches, including aliases.
        anchor_matches = {
            link["target"]
            for link in links
            if self.normalize(link["mention"]) == entity
        }

        # Otherwise, try matching the target title.
        target_matches = {
            link["target"]
            for link in links
            if self.normalize(link["target"]) == entity
        }

        if anchor_matches:
            candidates = anchor_matches
            method = "anchor_match"
        else:
            candidates = target_matches
            method = "title_match"

        # Avoid ambiguous mappings.
        if len(candidates) != 1:
            return None

        target = next(iter(candidates))

        return {
            "wikipedia_url": (
                "https://en.wikipedia.org/wiki/"
                + target
            ),
            "method": method,
        }



def entity_questions_link():
    linker = WikipediaEntityLinker(
        contact_email="burakayax@gmail.com"
    )

    entity_questions_path = "corpus/filtering/entity_questions/squad_validation.json"
    LABELS = ["PERSON", "ORGANIZATION", "LOCATION"]

    with open(entity_questions_path, "r", encoding="utf-8") as f:
        entity_questions = json.load(f)
        print(f"Loaded {len(entity_questions)} entity questions from {entity_questions_path}")

    for i, (question_id, question) in enumerate(entity_questions.items()):
        if question["label"] not in LABELS:
            continue

        title = get_by_question_id(question_id)["title"]
        entity_name = question["answer_entity"]
        link = linker.link_entity(title, entity_name)
        print(f"Question ID: {question_id}, Title: {title}, Entity: {entity_name}, Link: {link}")
        time.sleep(0.5)  # Sleep for 1 second to avoid hitting API rate limits


def standard_link():
    linker = WikipediaEntityLinker(
        contact_email="burakayax@gmail.com"
    )

    try:
        with open("evaluation/wikilinker/linked_entities.json", "r", encoding="utf-8") as f:
            link_results = json.load(f)
            last_question = list(link_results.values())[-1]
            last_question_id = list(link_results.keys())[-1]
            last_count = last_question["count"]
            last_index = last_question["index"]
            print(f"Resuming from last processed question ID: {last_question_id}")
    except FileNotFoundError:
        link_results = {}
        last_question_id = None
        last_count = 0
        last_index = 0
        print("No previously linked entities found. Starting fresh.")

    squad_triplets = triplets(Dataset.SQUAD_VALIDATION, skip_until_id=last_question_id)
    count = 0

    for i, (triplet, _) in enumerate(squad_triplets):
        title = get_by_question_id(triplet.question_id)["title"]
        entity_name = triplet.answer
        link = linker.link_entity(title, entity_name)
        if link is not None:
            count += 1
            link_results[triplet.question_id] = {
                "count": count + last_count,
                "index": i + last_index + 1,
                "question": triplet.question,
                "context": triplet.context,
                "answer": triplet.answer,
                "link": link
            }
            with open("evaluation/wikilinker/linked_entities.json", "w", encoding="utf-8") as f:
                json.dump(link_results, f, ensure_ascii=False, indent=4)
        print(f"Question ID: {triplet.question_id}, Title: {title}, Entity: {entity_name}, Link: {link}")
        time.sleep(0.5)  # Sleep for 1 second to avoid hitting API rate limits

        if count >= 10000:  # Limit to 200 successful links for demonstration
            break

    with open("evaluation/wikilinker/linked_entities.json", "w", encoding="utf-8") as f:
        json.dump(link_results, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(link_results)} linked entities to evaluation/wikilinker/linked_entities.json")

if __name__ == "__main__":
    standard_link()

