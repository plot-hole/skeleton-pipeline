from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import anthropic

from llm_deidentifier.config import DB_PATH, MIN_CONFIDENCE_TO_REPLACE
from llm_deidentifier.models.entities import ExtractedEntity, ReplacementSpan
from llm_deidentifier.pipeline.chunker import chunk_document
from llm_deidentifier.pipeline.extractor import extract_entities_from_chunk
from llm_deidentifier.pipeline.replacer import replace_entities
from llm_deidentifier.pipeline.resolver import EntityResolver
from llm_deidentifier.store.entity_store import EntityStore

logger = logging.getLogger(__name__)

CONFIDENCE_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


class DeIdentificationResult:
    """Holds the de-identified text and entity manifest."""

    def __init__(
        self,
        original_text: str,
        cleaned_text: str,
        manifest: list[dict[str, Any]],
    ) -> None:
        self.original_text = original_text
        self.cleaned_text = cleaned_text
        self.manifest = manifest

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.cleaned_text, encoding="utf-8")
        logger.info("Saved de-identified text to %s", path)

    def save_manifest(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved manifest to %s", path)


class DeIdentifier:
    """Main pipeline orchestrator."""

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self._client = anthropic.Anthropic()
        self._store = EntityStore(db_path)

    def process_file(self, file_path: str | Path) -> DeIdentificationResult:
        """Process a plain-text file through the full de-identification pipeline."""
        text = Path(file_path).read_text(encoding="utf-8")
        return self.process_text(text)

    def process_text(self, text: str) -> DeIdentificationResult:
        """Process raw text through the full pipeline."""
        # 1. Chunk
        chunks = chunk_document(text)
        logger.info("Split document into %d chunks", len(chunks))

        # 2. Extract entities from each chunk
        resolver = EntityResolver(self._client)
        all_entities: list[ExtractedEntity] = []

        for chunk in chunks:
            known = resolver.known_entities_for_prompt()
            entities = extract_entities_from_chunk(
                self._client, chunk, text, known,
            )
            logger.info(
                "Chunk %d: extracted %d entities", chunk.chunk_index, len(entities),
            )

            # Resolve immediately so next chunk gets updated known entities
            resolver.resolve(entities)
            all_entities.extend(entities)

        # 3. LLM-assisted ambiguous resolution
        resolver.resolve_ambiguous(all_entities)

        # 4. Assign tokens from the store
        for canon_key, resolved in resolver.canonical_map.items():
            token = self._store.get_or_create_token(
                resolved.canonical_text,
                resolved.entity_type,
                resolved.token_prefix,
            )
            resolver.set_token(canon_key, token)

        # Register aliases in the store
        for canon_key, resolved in resolver.canonical_map.items():
            for alias_norm, target_key in resolver._alias_index.items():
                if target_key == canon_key and alias_norm != canon_key:
                    self._store.register_alias(
                        alias_norm, resolved.canonical_text, resolved.entity_type,
                    )

        # 5. Build replacement spans
        min_conf = CONFIDENCE_ORDER.get(MIN_CONFIDENCE_TO_REPLACE, 1)
        spans: list[ReplacementSpan] = []
        manifest_entries: list[dict[str, Any]] = []

        for ent in all_entities:
            ent_conf = CONFIDENCE_ORDER.get(ent.confidence, 0)
            if ent_conf < min_conf:
                continue

            token = resolver.get_token_for_text(ent.text)
            if token is None:
                logger.warning("No token found for entity '%s'", ent.text)
                continue

            spans.append(ReplacementSpan(
                char_start=ent.char_start,
                char_end=ent.char_end,
                original_text=ent.text,
                replacement_token=token,
            ))

            manifest_entries.append({
                "original_text": ent.text,
                "replacement_token": token,
                "entity_type": ent.entity_type,
                "token_prefix": ent.token_prefix,
                "confidence": ent.confidence,
                "reasoning": ent.reasoning,
                "char_start": ent.char_start,
                "char_end": ent.char_end,
                "alias_of": ent.alias_of,
            })

        # 6. Replace
        cleaned = replace_entities(text, spans)

        return DeIdentificationResult(
            original_text=text,
            cleaned_text=cleaned,
            manifest=manifest_entries,
        )

    def close(self) -> None:
        self._store.close()


def process_folder(
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """Process every .txt file in input_dir and write results to output_dir."""
    from llm_deidentifier.config import INPUT_DIR, OUTPUT_DIR

    in_dir = Path(input_dir) if input_dir else INPUT_DIR
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    # Create folders if they don't exist
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(in_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {in_dir.resolve()}")
        print("Drop your plain-text documents there and run again.")
        return

    print(f"Found {len(txt_files)} file(s) in {in_dir.resolve()}")

    di = DeIdentifier()
    try:
        for file_path in txt_files:
            stem = file_path.stem
            output_path = out_dir / f"{stem}_clean.txt"
            manifest_path = out_dir / f"{stem}_manifest.json"

            print(f"\nProcessing: {file_path.name}")
            try:
                result = di.process_file(file_path)
                result.save(output_path)
                result.save_manifest(manifest_path)
                print(f"  -> {output_path.name}")
                print(f"  -> {manifest_path.name}")
            except Exception as exc:
                logger.error("Failed to process %s: %s", file_path.name, exc)
                print(f"  ERROR: {exc}")
    finally:
        di.close()

    print(f"\nDone. Results in {out_dir.resolve()}")


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Optional: pass custom input/output dirs as args
    input_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    process_folder(input_dir, output_dir)


if __name__ == "__main__":
    main()
