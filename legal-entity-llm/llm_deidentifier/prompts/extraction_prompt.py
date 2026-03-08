SYSTEM_PROMPT = """\
You are a legal document de-identification engine. Your job is to extract all personally \
identifiable information (PII) from legal document text and return structured JSON only.

For each entity found, determine:
- Its exact text as it appears in the document
- Its semantic role in the legal context
- The best token prefix for replacement

Return ONLY a JSON object. No explanation, no markdown, no preamble."""

USER_PROMPT_TEMPLATE = """\
Extract all PII entities from the following legal document excerpt.

KNOWN ENTITIES FROM PRIOR CHUNKS (use these to resolve aliases):
{known_entities_json}

DOCUMENT EXCERPT:
{chunk_text}

Return this exact JSON structure:
{{
  "entities": [
    {{
      "text": "exact text from document",
      "token_prefix": "PLAINTIFF | DEFENDANT | JUDGE | ATTORNEY | EXPERT_WITNESS | WITNESS | VICTIM | MINOR | LAW_FIRM | CORPORATION | COURT | GOVERNMENT_AGENCY | RURAL_{{STATE}}_ADDRESS | URBAN_{{STATE}}_ADDRESS | SUBURBAN_{{STATE}}_ADDRESS | INCIDENT_DATE | FILING_DATE | HEARING_DATE | BIRTH_DATE | SETTLEMENT_AMOUNT | DAMAGES_AMOUNT | CASE_NUMBER | SSN | PHONE | EMAIL | EXHIBIT | STATUTE | PERSON | ORGANIZATION | ADDRESS | DATE | DOLLAR_AMOUNT",
      "entity_type": "PERSON | ORG | ADDRESS | DATE | MONEY | ID",
      "alias_of": "canonical entity text this is an alias for, or null",
      "reasoning": "one sentence explaining role inference",
      "confidence": "HIGH | MEDIUM | LOW"
    }}
  ]
}}

Rules:
- text must be the EXACT string from the document, character-for-character
- For addresses, classify as RURAL/URBAN/SUBURBAN based on context clues (rural route, farm, township vs city, downtown, metropolitan)
- For persons, infer role from surrounding context (how they are described, referenced, their actions)
- NEVER extract pronouns (he, she, they, his, her, him, them, this, that) — these are NOT PII
- NEVER extract generic legal role descriptors used as labels: "Plaintiff", "Defendant", "the plaintiff", "the defendant", "Plaintiffs", "Defendants" — these describe roles, not identities
- Only extract PROPER NAMES of people, not role words. "Plaintiff" is a role label. "John Smith" is PII.
- Mark honorifics/partial names ("Mr. Smith", "Dr. Jones") as aliases of the full proper name
- CRITICAL: Extract EVERY occurrence of an entity name, including shortened/informal variants. If "Northwest Logistics Solutions, LLC" is a party, also extract "Northwest Logistics Solutions", "Northwest Logistics", and "Northwest" (when referring to that company) as separate entities with alias_of pointing to the full formal name. Every surface form that appears in the text must be its own entity entry.
- For organizations: extract the full formal name AND every shortened form used in the text. Each variant is a separate entity with its own "text" field and alias_of pointing to the canonical full name.
- If role is genuinely ambiguous, use the generic type (PERSON, ORGANIZATION, ADDRESS)
- Return LOW confidence when role is inferred with limited context"""

RETRY_SUFFIX = "\n\nReturn only raw JSON, no other text."

RESOLUTION_PROMPT = """\
In a legal document, are these two entities the same person or thing?
Entity A: "{entity_a}" ({type_a})
Entity B: "{entity_b}" ({type_b})
Answer YES or NO only."""
