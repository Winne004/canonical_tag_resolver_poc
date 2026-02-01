from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="Search query string"
    )
    k: int = Field(default=5, ge=1, le=100, description="Number of results to return")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()


class SearchResult(BaseModel):
    """Individual search result."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    results: list[SearchResult]
    count: int = Field(..., ge=0)


class ResolveRequest(BaseModel):
    """Request model for resolve endpoint."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="Canonical key to resolve"
    )
    similarity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    k: int = Field(
        default=5, ge=1, le=100, description="Number of suggestions to return"
    )
    enable_llm_review: bool = Field(
        default=False,
        description="Enable LLM review to determine if tags can be auto-merged",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace, strip and normalize to lowercase."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip().lower()


class CanonicalKeySuggestion(BaseModel):
    """Individual canonical key suggestion."""

    canonical_key: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    llm_review: dict[str, Any] | None = Field(
        default=None, description="LLM review decision for auto-merge vs human review"
    )


class ResolveResponseExact(BaseModel):
    """Response model for exact match in resolve endpoint."""

    found: bool = Field(default=True)
    canonical_key: str
    match_type: str = Field(default="exact")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResolveResponseSuggestions(BaseModel):
    """Response model for suggestions in resolve endpoint."""

    found: bool = Field(default=False)
    match_type: str = Field(default="suggestions")
    query: str
    suggestions: list[CanonicalKeySuggestion]
    count: int = Field(..., ge=0)


class IngestRequest(BaseModel):
    """Request model for ingest endpoint."""

    topics: list[str] = Field(
        ..., min_length=1, max_length=1000, description="List of topics to ingest"
    )

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        """Validate topics are not empty, strip whitespace, normalize to lowercase, and remove duplicates."""
        if not v:
            raise ValueError("Topics list cannot be empty")

        seen = set()
        unique_topics = []
        for topic in v:
            if not isinstance(topic, str):
                raise ValueError("All topics must be strings")

            topic_stripped = topic.strip().lower()
            if not topic_stripped:
                raise ValueError("Topics cannot be empty or only whitespace")

            if len(topic_stripped) > 500:
                raise ValueError(
                    f"Topic too long (max 500 characters): {topic_stripped[:50]}..."
                )

            if topic_stripped not in seen:
                seen.add(topic_stripped)
                unique_topics.append(topic_stripped)

        return unique_topics


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""

    message: str
    count: int = Field(..., ge=0)


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    details: str | None = None
