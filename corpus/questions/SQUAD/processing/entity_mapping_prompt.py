ENTITY_MAPPING_PROMPT = """
You are an entity-mapping generator for a QA dataset augmentation task.

Your task:
Given one JSON object with:
- context: a passage
- questions: questions about the passage
- answers: gold answers

Return a JSON object with one field:

{
  "entities": [
    {"name": "...", "mapping": "..."}
  ]
}

The goal is to create replacement mappings for controlled counterfactual QA experiments.

Rules:

1. Extract literal surface strings that should be replaced.
   Include proper nouns, person names, organizations, companies, product names, work titles, locations, events, identifiers, acronyms, dates, years, numbers, money amounts, quantities, percentages, and domain-specific named terms.

2. The "name" field must be an exact substring appearing in the input.
   Do not normalize it.
   Do not invent a name that is not literally present.
   If both "Beyoncé" and "Beyonce" appear, include both separately.
   If both "June 2013" and "2013" appear and both may be answered directly, include both separately.

3. The "mapping" must be the same semantic type as the original.
   Examples:
   - person -> different fictional person name
   - company -> different fictional company name
   - product -> different fictional product name
   - song/movie/book/game title -> different fictional title
   - place -> different fictional place of the same kind
   - date -> different date
   - year -> different year
   - money amount -> different money amount
   - count/quantity -> different count/quantity
   - acronym/identifier -> different acronym/identifier

4. Preserve format and grammatical compatibility.
   - "$100 million" should map to something like "$240 million"
   - "70 staff" should map to something like "85 staff"
   - "June 2013" should map to something like "April 2014"
   - "NDtv" should map to something like "QRtv"
   - "WSND-FM" should map to something like "KVRP-FM"
   - "Starpower: Beyoncé" should map to another video-game title with a similar structure

5. Prefer fictional replacements.
   Avoid replacing entities with famous real entities.
   This is important because the QA model should not be able to rely on world knowledge.

6. Keep mappings consistent inside this object.
   If a larger string contains a smaller mapped string, the mappings should be compatible.
   Example:
   - "Beyoncé" -> "Amara Vale"
   - "Starpower: Beyoncé" -> "Starpower: Amara Vale"

7. Include answer strings when they are replaceable entities, numbers, dates, identifiers, or domain-specific names.
   For example, if an answer is "GateFive", include a mapping for "GateFive".
   If an answer is "18", include a mapping for "18" only if replacing that number would still keep the passage coherent.

8. Do not include ordinary common nouns unless they function as domain-specific terms or answer-critical identifiers.
   Good to include: "Nintendo DS", "iTunes Store", "Great Sichuan earthquake", "Level II"
   Usually do not include: "lawyers", "staff", "financial backers", "music career"

9. Do not explain your choices.
   Do not output Markdown.
   Do not wrap the JSON in a code block.
   Return valid JSON only.

10. Output order:
   Sort entities by descending length of "name".
   This helps safe longest-first string replacement.

Input JSON:
{{INPUT_JSON}}
"""