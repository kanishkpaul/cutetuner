from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class InputMode(StrEnum):
    FULL_MIX = "full_mix"
    VOCAL_BACKING = "vocal_backing"
    VOCAL_ONLY = "vocal_only"


class ProjectSummary(BaseModel):
    id: str
    name: str
    mode: InputMode
    status: str
    created_at: str
    updated_at: str
    has_report: bool = False
    has_outputs: bool = False


class Metric(BaseModel):
    value: float | str | None
    unit: str = ""
    confidence: float = Field(ge=0, le=1)
    source: str


class NoteEvent(BaseModel):
    start: float
    end: float
    midi: float
    target_midi: float | None = None
    confidence: float


class ChordEvent(BaseModel):
    start: float
    end: float
    label: str
    confidence: float


class AnalysisReport(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    key: str | None
    scale: str | None
    key_confidence: float
    key_alternatives: list[str] = Field(default_factory=list)
    technical: dict[str, Metric]
    vocal: dict[str, Metric]
    semantic_descriptors: list[str]
    naturalness: Metric
    notes: list[NoteEvent]
    chords: list[ChordEvent]
    what_i_hear: list[str]
    warnings: list[str]
    analyzer_status: dict[str, str]


class ProtectedRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end <= self.start:
            raise ValueError("protected range must end after it starts")
        return self


class CreativeBrief(BaseModel):
    vibe: str = Field(min_length=2, max_length=500)
    delivery: str = Field(default="honest and controlled", max_length=300)
    lyrics: str | None = Field(default=None, max_length=20_000)
    reference_description: str | None = Field(default=None, max_length=500)
    tuning_intent: Literal["transparent", "balanced", "obvious"] = "transparent"
    protected_ranges: list[ProtectedRange] = Field(default_factory=list)


class TuningControls(BaseModel):
    key: str | None = None
    scale: Literal["major", "minor", "chromatic"] | None = None
    strength: float = Field(default=0.58, ge=0, le=1)
    retune_ms: float = Field(default=105, ge=5, le=300)
    max_correction_cents: float = Field(default=180, ge=10, le=600)
    vibrato_preservation: float = Field(default=0.85, ge=0, le=1)
    formant_shift: float = Field(default=0, ge=-2, le=2)
    breath_protection: float = Field(default=0.9, ge=0, le=1)
    eq_amount: float = Field(default=0.35, ge=0, le=1)
    deesser_amount: float = Field(default=0.3, ge=0, le=1)
    compression_amount: float = Field(default=0.35, ge=0, le=1)
    ambience_amount: float = Field(default=0.12, ge=0, le=1)
    vocal_gain_db: float = Field(default=0, ge=-12, le=12)


class TuningPlan(BaseModel):
    rationale: list[str]
    controls: TuningControls
    notes: list[NoteEvent]
    chords: list[ChordEvent]
    protected_ranges: list[ProtectedRange]
    source: str


class OutputFile(BaseModel):
    kind: Literal[
        "corrected_vocal",
        "finished_mix",
        "preview_natural",
        "preview_recommended",
        "preview_tighter",
    ]
    filename: str
    media_type: str = "audio/mpeg"


class ProjectDetail(ProjectSummary):
    report: AnalysisReport | None = None
    brief: CreativeBrief | None = None
    plan: TuningPlan | None = None
    outputs: list[OutputFile] = Field(default_factory=list)


class RenderRequest(BaseModel):
    controls: TuningControls | None = None


class LocalLLMConfig(BaseModel):
    endpoint: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=300)


class RatingRequest(BaseModel):
    winner: Literal["natural", "recommended", "tighter", "original"]
    comment: str | None = Field(default=None, max_length=1_000)
    context: dict[str, float | str | bool | None] = Field(default_factory=dict)


class JobState(BaseModel):
    id: str
    project_id: str
    kind: Literal["analysis", "preview", "render"]
    stage: str
    progress: float = Field(ge=0, le=1)
    status: Literal["queued", "running", "complete", "failed", "cancelled"]
    message: str = ""
    error: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
