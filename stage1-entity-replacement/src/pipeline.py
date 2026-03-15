"""Main pipeline orchestrator.

Runs all Stage 1 components in sequence:
  1. Privilege check (early exit)
  2. Co-reference resolution
  3. Replacement
  4. Relationship graph building
  5. Validation (second-pass NER)
"""

from __future__ import annotations

from datetime import date

from src.coreference import CoReferenceEngine
from src.relationships import RelationshipGraphBuilder
from src.replacement import ReplacementEngine
from src.schemas import (
    ClassificationMetadata,
    PipelineOutput,
    PrivilegeHaltError,
    TokenMappingTable,
    ValidationReport,
)
from src.validation import Validator


class Pipeline:
    """Stage 1 entity replacement pipeline."""

    def __init__(self, reference_date: date | None = None) -> None:
        self.reference_date = reference_date

    def run(
        self,
        document_text: str,
        metadata: ClassificationMetadata,
    ) -> PipelineOutput:
        """Execute the full pipeline and return the de-identified output.

        Raises
        ------
        PrivilegeHaltError
            If the document contains a PRIVILEGE_INDICATOR element.
        """
        # ---------------------------------------------------------------
        # Step 0: Privilege check (early exit)
        # ---------------------------------------------------------------
        if metadata.summary.privilege_flag:
            raise PrivilegeHaltError(
                f"Document {metadata.document_id} flagged with privilege_flag"
            )
        for element in metadata.elements:
            if element.category == "PRIVILEGE_INDICATOR":
                raise PrivilegeHaltError(
                    f"Element {element.element_id} is PRIVILEGE_INDICATOR"
                )

        # ---------------------------------------------------------------
        # Step 1: Co-reference resolution
        # ---------------------------------------------------------------
        coref_engine = CoReferenceEngine(metadata.elements)
        entity_groups = coref_engine.resolve()

        # ---------------------------------------------------------------
        # Step 2: Replacement
        # ---------------------------------------------------------------
        replacement_engine = ReplacementEngine(
            document_text,
            metadata,
            entity_groups,
            reference_date=self.reference_date,
        )
        replaced_text, token_mappings = replacement_engine.replace_all()

        # ---------------------------------------------------------------
        # Step 3: Relationship graph
        # ---------------------------------------------------------------
        graph_builder = RelationshipGraphBuilder(metadata.elements, entity_groups)
        relationship_graph = graph_builder.build()

        # ---------------------------------------------------------------
        # Step 4: Validation (second-pass NER)
        # ---------------------------------------------------------------
        validator = Validator()
        validation_result = validator.validate(replaced_text)

        # ---------------------------------------------------------------
        # Step 5: Build output
        # ---------------------------------------------------------------
        # Count stats
        elements_replaced = sum(
            1 for e in metadata.elements if e.category != "BOILERPLATE"
        )
        elements_passed = sum(
            1 for e in metadata.elements if e.category == "BOILERPLATE"
        )
        co_references_resolved = sum(
            g.occurrence_count - 1 for g in entity_groups if g.occurrence_count > 1
        )

        validation_report = ValidationReport(
            document_id=metadata.document_id,
            validation_passed=validation_result.passed,
            ner_entities_detected=validation_result.entities_detected,
            ner_entities_allowed=validation_result.entities_allowed,
            ner_entities_flagged=validation_result.entities_flagged,
            elements_replaced=elements_replaced,
            elements_passed_through=elements_passed,
            co_references_resolved=co_references_resolved,
            unique_tokens_assigned=len(entity_groups),
        )

        return PipelineOutput(
            deidentified_text=replaced_text,
            token_mapping=TokenMappingTable(
                document_id=metadata.document_id,
                tokens=token_mappings,
            ),
            relationship_graph=relationship_graph,
            validation_report=validation_report,
        )
