from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphEntity(BaseModel):
    type: Literal["cve", "cwe", "unknown"] = "unknown"
    id: str = ""


class Citation(BaseModel):
    citation_id: str
    source_type: str
    entity_id: str
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    cve_id: str
    likelihood: float
    evidence_tier: str
    rel_type: str = ""
    signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    inferred_from: list[str] = Field(default_factory=list)


class ConfidenceSummary(BaseModel):
    overall: float
    rationale: str


class HITLDecision(BaseModel):
    required: bool
    reasons: list[str] = Field(default_factory=list)


class GraphRAGQueryRequest(BaseModel):
    query: str
    entity: GraphEntity | None = None
    top_k: int = Field(default=12, ge=1, le=25)


class GraphRAGAgentResponse(BaseModel):
    status: Literal["ok", "needs_human_review", "error"] = "ok"
    query: str
    entity: GraphEntity = Field(default_factory=GraphEntity)
    direct_evidence: list[EvidenceItem] = Field(default_factory=list)
    inferred_candidates: list[EvidenceItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence_summary: ConfidenceSummary = Field(
        default_factory=lambda: ConfidenceSummary(overall=0.0, rationale="")
    )
    hitl: HITLDecision = Field(default_factory=lambda: HITLDecision(required=False))
    recommended_actions: list[str] = Field(default_factory=list)
    error: str | None = None


def dump_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

