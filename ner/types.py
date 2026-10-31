from dataclasses import dataclass

@dataclass
class NamedEntity:
    name: str
    label: str
    start_char: int | None = None
    end_char: int | None = None