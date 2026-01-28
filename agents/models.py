from pydantic import BaseModel, Field, field_validator


class Finding(BaseModel):
    """A single research finding with claim, source, and details."""

    claim: str = Field(..., min_length=1, description="The factual claim")
    source: str = Field(..., min_length=1, description="URL or source reference")
    details: str = Field(..., min_length=1, description="Supporting details")


class ResearchResult(BaseModel):
    """Results from researching a single subtask."""

    subtask: str = Field(..., min_length=1, description="The research subtask")
    findings: list[Finding] = Field(
        default_factory=list, description="List of research findings"
    )

    @field_validator("findings")
    @classmethod
    def validate_findings(cls, v: list[Finding]) -> list[Finding]:
        """Ensure at least one finding is present."""
        if len(v) == 0:
            raise ValueError("Must have at least one finding")

        return v


class CoordinatorResponse(BaseModel):
    """Response from the coordinator agent breaking down a query into subtasks."""

    subtasks: list[str] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="2-4 research subtasks to investigate",
    )

    @field_validator("subtasks")
    @classmethod
    def validate_subtasks(cls, v: list[str]) -> list[str]:
        """Ensure all subtasks are non-empty strings."""
        if not all(isinstance(task, str) and len(task.strip()) > 0 for task in v):
            raise ValueError("All subtasks must be non-empty strings")

        return [task.strip() for task in v]


class SynthesisSection(BaseModel):
    """A section in the synthesized research report."""

    title: str = Field(..., min_length=1, description="Section title")
    content: str = Field(..., min_length=1, description="Section content")
    sources: list[str] = Field(
        default_factory=list, description="Source URLs cited in this section"
    )


class SynthesizedReport(BaseModel):
    """Synthesized research report combining all findings."""

    summary: str = Field(..., min_length=1, description="Executive summary")
    sections: list[SynthesisSection] = Field(
        ..., min_length=1, description="Report sections organized by theme"
    )
    key_insights: list[str] = Field(
        default_factory=list, description="Key takeaways from the research"
    )

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v: list[SynthesisSection]) -> list[SynthesisSection]:
        """Ensure at least one section is present."""
        if len(v) == 0:
            raise ValueError("Must have at least one section")

        return v


class CriticIssue(BaseModel):
    """An issue identified by the critic agent."""

    type: str = Field(..., min_length=1, description="Issue type")
    description: str = Field(..., min_length=1, description="Issue description")
    location: str = Field(..., min_length=1, description="Where the issue was found")
    severity: str = Field(..., description="Issue severity (low, medium, high)")

    @property
    def formatted_type(self) -> str:
        """Human-readable issue type."""
        return self.type.replace("_", " ").title()


class CriticReview(BaseModel):
    """Review from the critic agent validating research quality."""

    overall_quality: str = Field(
        ..., min_length=1, description="Overall quality assessment"
    )
    issues: list[CriticIssue] = Field(
        default_factory=list, description="Issues found in the report"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )
    needs_more_research: bool = Field(
        ..., description="Whether additional research is needed"
    )


class WorkflowResult(BaseModel):
    """Complete result from the research workflow."""

    query: str = Field(..., min_length=1, description="Original research query")
    subtasks: list[str] = Field(..., description="Research subtasks identified")
    research_results: list[ResearchResult] = Field(
        ..., description="Results from each subtask"
    )
    synthesis: SynthesizedReport = Field(..., description="Synthesized report")
    critique: CriticReview = Field(..., description="Quality review")


# Tool Models

class SearchResult(BaseModel):
    """A single search result from web search tool."""

    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    content: str = Field(..., description="Content snippet from the search result")
    score: float = Field(default=0.0, description="Relevance score of the result")


class ToolSchema(BaseModel):
    """Schema definition for a tool that can be used by agents."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description for LLM")
    input_schema: dict = Field(..., description="JSON schema for tool inputs")

    def to_dict(self) -> dict:
        """Convert to dictionary format expected by Anthropic API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
