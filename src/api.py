from http import HTTPStatus
from typing import Any

from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, CORSConfig
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext
from botocore.exceptions import ClientError
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from src.models import (
    CanonicalKeySuggestion,
    IngestRequest,
    IngestResponse,
    ResolveRequest,
    ResolveResponseExact,
    ResolveResponseSuggestions,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.shared import (
    get_azure_settings,
    get_chat_model,
    get_dynamodb_table,
    get_vector_store,
)

# Initialize Powertools
logger = Logger()
tracer = Tracer()
metrics = Metrics()
cors_config = CORSConfig(allow_origin="*", max_age=300)

app = APIGatewayRestResolver(
    enable_validation=True,
    response_validation_error_http_code=HTTPStatus.INTERNAL_SERVER_ERROR,
    cors=cors_config,
)

app.enable_swagger()


def review_tag_merge_with_llm(
    query_tag: str,
    canonical_tag: str,
    similarity_score: float,
) -> dict[str, Any]:
    """Use LLM to review if a tag can be automatically merged or requires human review.

    Args:
        query_tag: The original query tag
        canonical_tag: The suggested canonical tag
        similarity_score: The semantic similarity score

    Returns:
        Dictionary with review decision including:
        - can_auto_merge: bool indicating if auto-merge is recommended
        - confidence: str (high/medium/low)
        - reasoning: str explaining the decision
        - requires_human_review: bool

    """
    try:
        llm = get_chat_model()

        parser = JsonOutputParser()

        prompt_template = PromptTemplate(
            template="""You are a tag management expert. Review if the following tags can be automatically merged.

Query Tag: "{query_tag}"
Canonical Tag: "{canonical_tag}"
Semantic Similarity Score: {similarity_score}

Analyze if these tags can be automatically merged based on:
1. Semantic equivalence (are they truly the same concept?)
2. Similarity score (higher = more confident)
3. Risk of incorrect merging

Guidelines:

- AUTO_MERGE with HIGH confidence if similarity >= 0.8 and tags are very similar
- HUMAN_REVIEW if similarity < 0.80 or there's any ambiguity
- HUMAN_REVIEW with LOW confidence if tags might have different meanings

ignore case sensitivity or minor variations

{format_instructions}

Provide your response as a JSON object with these fields:
- decision: "AUTO_MERGE" or "HUMAN_REVIEW"
- confidence: "high", "medium", or "low"
- reasoning: Brief explanation of your decision""",
            input_variables=["query_tag", "canonical_tag", "similarity_score"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt_template | llm | parser

        result = chain.invoke(
            {
                "query_tag": query_tag.lower().strip(),
                "canonical_tag": canonical_tag.lower().strip(),
                "similarity_score": f"{similarity_score:.3f}",
            },
        )

        decision = result.get("decision", "HUMAN_REVIEW")
        confidence = result.get("confidence", "low").lower()
        reasoning = result.get("reasoning", "No reasoning provided")
        can_auto_merge = decision == "AUTO_MERGE"

        return {
            "can_auto_merge": can_auto_merge,
            "confidence": confidence,
            "reasoning": reasoning,
            "requires_human_review": not can_auto_merge,
            "similarity_score": similarity_score,
        }

    except Exception as e:
        logger.warning(f"LLM review failed: {e}")

        return {
            "can_auto_merge": False,
            "confidence": "low",
            "reasoning": f"LLM review failed: {e!s}",
            "requires_human_review": True,
            "similarity_score": similarity_score,
        }


@app.post("/search")
@tracer.capture_method
def search_topics(request: SearchRequest) -> dict[str, Any]:
    """Search for canonical topics using semantic similarity."""
    logger.info(f"Searching for: {request.query}")

    try:
        azure_settings = get_azure_settings()
        vector_store = get_vector_store()

        search_results = vector_store.similarity_search_with_score(
            request.query,
            k=request.k,
            embedder_name=azure_settings.embedding_model,
        )

        results = [
            SearchResult(
                content=doc[0].page_content,
                metadata=doc[0].metadata,
                similarity_score=float(doc[1]),
            )
            for doc in search_results
        ]

        response = SearchResponse(results=results, count=len(results))

        logger.info(f"Found {response.count} results")
        metrics.add_metric(name="SearchSuccess", unit=MetricUnit.Count, value=1)
        metrics.add_metric(
            name="ResultsReturned",
            unit=MetricUnit.Count,
            value=response.count,
        )

        return response.model_dump()

    except Exception:
        logger.exception("Error during search")
        metrics.add_metric(name="SearchErrors", unit=MetricUnit.Count, value=1)
        raise


@app.post("/resolve")
@tracer.capture_method
def resolve_canonical_key(request: ResolveRequest) -> dict[str, Any]:
    """Resolve a canonical key. Check DynamoDB first, then suggest similar keys from vector store if not found."""
    logger.info(f"Resolving canonical key for: {request.query}")

    try:
        table = get_dynamodb_table()

        try:
            response_db = table.get_item(Key={"canonical_key": request.query})
            if "Item" in response_db:
                logger.info(f"Found exact match in DynamoDB: {request.query}")
                metrics.add_metric(
                    name="ExactMatchFound",
                    unit=MetricUnit.Count,
                    value=1,
                )
                response = ResolveResponseExact(
                    canonical_key=response_db["Item"]["canonical_key"],
                    metadata=response_db["Item"].get("metadata", {}),
                )
                return response.model_dump()
        except ClientError as e:
            logger.error(f"Error querying DynamoDB: {e}")

        logger.info("No exact match found, searching vector store for similar keys")
        azure_settings = get_azure_settings()
        vector_store = get_vector_store()

        search_results = vector_store.similarity_search_with_score(
            request.query,
            k=request.k,
            embedder_name=azure_settings.embedding_model,
        )

        suggestions = []
        for doc, score in search_results:
            llm_review = None

            if request.enable_llm_review:
                logger.info(
                    f"Performing LLM review for '{request.query}' -> '{doc.page_content}'",
                )
                llm_review = review_tag_merge_with_llm(
                    query_tag=request.query,
                    canonical_tag=doc.page_content,
                    similarity_score=float(score),
                )
                metrics.add_metric(
                    name="LLMReviewsPerformed",
                    unit=MetricUnit.Count,
                    value=1,
                )

            suggestions.append(
                CanonicalKeySuggestion(
                    canonical_key=doc.page_content,
                    similarity_score=float(score),
                    metadata=doc.metadata,
                    llm_review=llm_review,
                ),
            )

        response = ResolveResponseSuggestions(
            query=request.query,
            suggestions=suggestions,
            count=len(suggestions),
        )

        logger.info(f"Found {response.count} similar keys")
        metrics.add_metric(name="SimilarKeysFound", unit=MetricUnit.Count, value=1)
        metrics.add_metric(
            name="SuggestionsReturned",
            unit=MetricUnit.Count,
            value=response.count,
        )

        return response.model_dump()

    except Exception:
        logger.exception("Error during canonical key resolution")
        metrics.add_metric(name="ResolveErrors", unit=MetricUnit.Count, value=1)
        raise


@app.post("/ingest")
@tracer.capture_method
def ingest_topics(request: IngestRequest) -> dict[str, Any]:
    """Ingest an array of topics into the Meilisearch vector store and DynamoDB."""
    logger.info(f"Ingesting {len(request.topics)} topics into index and DynamoDB")

    try:
        table = get_dynamodb_table()

        with table.batch_writer() as batch:
            for topic in request.topics:
                batch.put_item(
                    Item={
                        "canonical_key": topic,
                        "metadata": {},
                    },
                )

        response = IngestResponse(
            message=f"Successfully ingested {len(request.topics)} topics",
            count=len(request.topics),
        )

        logger.info(f"Successfully ingested {len(request.topics)} topics")
        metrics.add_metric(name="TopicsIngested", unit=MetricUnit.Count, value=1)
        metrics.add_metric(
            name="TopicsAdded",
            unit=MetricUnit.Count,
            value=len(request.topics),
        )

        return response.model_dump()

    except Exception:
        logger.exception("Error during topic ingestion")
        metrics.add_metric(name="IngestionErrors", unit=MetricUnit.Count, value=1)
        raise


@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    """Main Lambda handler for search requests."""
    return app.resolve(event, context)
