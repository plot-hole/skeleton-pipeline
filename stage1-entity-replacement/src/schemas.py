"""Pydantic models for all input/output schemas in the entity replacement pipeline."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input models (from Cyera classification metadata)
# ---------------------------------------------------------------------------

class ElementContext(BaseModel):
    data_subject_role: Literal[
        "PARTY_PRIMARY",
        "PARTY_SECONDARY",
        "MINOR_DEPENDENT",
        "EMPLOYER",
        "LEGAL_REPRESENTATIVE",
    ]
    jurisdiction: str = ""
    applicable_regulations: list[str] = []
    industry: str | None = None


class ClassifiedElement(BaseModel):
    element_id: str
    category: Literal[
        "DIRECT_IDENTIFIER",
        "QUASI_IDENTIFIER",
        "FINANCIAL_DATA",
        "HEALTH_PROTECTED",
        "MINOR_INFORMATION",
        "PRIVILEGE_INDICATOR",
        "BOILERPLATE",
    ]
    subcategory: str
    sensitivity: Literal["CRITICAL", "HIGH", "LOW"]
    byte_offset_start: int
    byte_offset_end: int
    original_text: str
    context: ElementContext


class ClassificationSummary(BaseModel):
    total_elements: int
    critical_count: int
    high_count: int
    low_count: int
    privilege_flag: bool
    minor_data_present: bool


class ClassificationMetadata(BaseModel):
    document_id: str
    tenant_id: str
    classification_timestamp: str
    classifier_version: str
    elements: list[ClassifiedElement]
    summary: ClassificationSummary


# ---------------------------------------------------------------------------
# Internal models (co-reference engine output)
# ---------------------------------------------------------------------------

class EntityGroup(BaseModel):
    token: str  # e.g. "[PARTY_A]"
    category: str  # e.g. "DIRECT_IDENTIFIER"
    element_ids: list[str]
    data_subject_role: str
    occurrence_count: int


# ---------------------------------------------------------------------------
# Output models — Token Mapping
# ---------------------------------------------------------------------------

class TokenMapping(BaseModel):
    token: str
    category: str
    element_ids: list[str]
    occurrence_count: int
    data_subject_role: str | None = None
    generalization_level: str | None = None
    original_jurisdiction: str | None = None


class TokenMappingTable(BaseModel):
    document_id: str
    tokens: list[TokenMapping]


# ---------------------------------------------------------------------------
# Output models — Relationship Graph (discriminated union)
# ---------------------------------------------------------------------------

class CohabitationRelationship(BaseModel):
    type: Literal["COHABITATION"] = "COHABITATION"
    entities: list[str]
    location: str


class ParentChildRelationship(BaseModel):
    type: Literal["PARENT_CHILD"] = "PARENT_CHILD"
    entity: str
    dependents: list[str]


class EmploymentRelationship(BaseModel):
    type: Literal["EMPLOYMENT"] = "EMPLOYMENT"
    entity: str
    employer: str
    industry: str


Relationship = Annotated[
    CohabitationRelationship | ParentChildRelationship | EmploymentRelationship,
    Field(discriminator="type"),
]


class RelationshipGraph(BaseModel):
    relationships: list[
        Annotated[
            CohabitationRelationship | ParentChildRelationship | EmploymentRelationship,
            Field(discriminator="type"),
        ]
    ] = []


# ---------------------------------------------------------------------------
# Output models — Validation Report
# ---------------------------------------------------------------------------

class ValidationReport(BaseModel):
    document_id: str
    validation_passed: bool
    ner_entities_detected: int
    ner_entities_allowed: int
    ner_entities_flagged: int
    elements_replaced: int
    elements_passed_through: int
    co_references_resolved: int
    unique_tokens_assigned: int


# ---------------------------------------------------------------------------
# Pipeline output (all four outputs bundled)
# ---------------------------------------------------------------------------

class PipelineOutput(BaseModel):
    deidentified_text: str
    token_mapping: TokenMappingTable
    relationship_graph: RelationshipGraph
    validation_report: ValidationReport


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PrivilegeHaltError(Exception):
    """Raised when a PRIVILEGE_INDICATOR element is found in the document."""
