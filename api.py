from functools import lru_cache
from http import HTTPStatus
from typing import Any

import boto3
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.event_handler import APIGatewayRestResolver, CORSConfig
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext
from botocore.exceptions import ClientError
from langchain_community.vectorstores import Meilisearch
from langchain_openai import AzureOpenAIEmbeddings
from meilisearch import Client as MeiliClient

from models import (
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
from settings import AzureOpenAISettings, DynamoDBSettings, MeilisearchSettings

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


@lru_cache(maxsize=1)
def get_dynamodb_table():
    """Get or create DynamoDB table resource (cached)."""
    dynamodb_settings = get_dynamodb_settings()
    dynamodb = boto3.resource("dynamodb")
    return dynamodb.Table(dynamodb_settings.canonical_tags_table)  # pyright: ignore[reportAttributeAccessIssue]


@lru_cache(maxsize=1)
def get_azure_settings() -> AzureOpenAISettings:
    """Get or create Azure OpenAI settings (cached)."""
    return AzureOpenAISettings()  # pyright: ignore[reportCallIssue]


@lru_cache(maxsize=1)
def get_meili_settings() -> MeilisearchSettings:
    """Get or create Meilisearch settings (cached)."""
    return MeilisearchSettings()  # pyright: ignore[reportCallIssue]


@lru_cache(maxsize=1)
def get_dynamodb_settings() -> DynamoDBSettings:
    """Get or create DynamoDB settings (cached)."""
    return DynamoDBSettings()  # pyright: ignore[reportCallIssue]


@lru_cache(maxsize=1)
def get_meili_client() -> MeiliClient:
    """Get or create Meilisearch client (cached)."""
    meili_settings = get_meili_settings()
    return MeiliClient(
        meili_settings.meili_url,
        meili_settings.meli_master_key.get_secret_value(),
    )


@lru_cache(maxsize=1)
def get_embeddings() -> AzureOpenAIEmbeddings:
    """Get or create Azure OpenAI embeddings (cached)."""
    azure_settings = get_azure_settings()
    return AzureOpenAIEmbeddings(
        model=azure_settings.embedding_model,
        azure_endpoint=azure_settings.endpoint,
        api_key=azure_settings.azure_open_ai_api_key,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Meilisearch:
    """Get or create Meilisearch vector store (cached)."""
    meili_settings = get_meili_settings()
    meili_client = get_meili_client()
    embeddings = get_embeddings()

    return Meilisearch(
        client=meili_client,
        index_name=meili_settings.index_name,
        embedding=embeddings,
    )


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

        suggestions = [
            CanonicalKeySuggestion(
                canonical_key=doc.page_content,
                similarity_score=float(score),
                metadata=doc.metadata,
            )
            for doc, score in search_results
        ]

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
        azure_settings = get_azure_settings()
        table = get_dynamodb_table()

        embedders = {
            f"{azure_settings.embedding_model}": {
                "source": "userProvided",
                "dimensions": azure_settings.dimensions,
            },
        }

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
