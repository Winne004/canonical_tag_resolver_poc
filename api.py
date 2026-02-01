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
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from meilisearch import Client as MeiliClient

from src.models import (
    CanonicalKeySuggestion,
    CreateAliasesRequest,
    CreateAliasesResponse,
    CreateAliasRequest,
    CreateAliasResponse,
    DeleteAliasRequest,
    DeleteAliasResponse,
    IngestRequest,
    IngestResponse,
    ListAliasesResponse,
    ResolveRequest,
    ResolveResponseExact,
    ResolveResponseSuggestions,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.settings import AzureOpenAISettings, DynamoDBSettings, MeilisearchSettings

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
def get_chat_model() -> AzureChatOpenAI:
    """Get or create Azure OpenAI chat model (cached)."""
    azure_settings = get_azure_settings()
    return AzureChatOpenAI(
        azure_deployment=azure_settings.llm_deployment,  # You may want to add this to settings
        azure_endpoint=azure_settings.endpoint,
        api_key=azure_settings.azure_open_ai_api_key,
        api_version="2024-02-15-preview",
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Meilisearch:
    """Get or create Meilisearch vector store (cached).

    This creates the vector store once with embedders config.
    The embedders will be set only on first initialization.
    """
    meili_settings = get_meili_settings()
    meili_client = get_meili_client()
    embeddings = get_embeddings()
    azure_settings = get_azure_settings()

    embedders = {
        azure_settings.embedding_model: {
            "source": "userProvided",
            "dimensions": azure_settings.dimensions,
        },
    }

    return Meilisearch(
        client=meili_client,
        index_name=meili_settings.index_name,
        embedding=embeddings,
        embedders=embedders,
    )


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
    """Resolve a canonical key. Check DynamoDB for exact match or alias, then suggest similar keys from vector store if not found."""
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
                    is_alias=False,
                )
                return response.model_dump()
        except ClientError as e:
            logger.error(f"Error querying DynamoDB: {e}")

        try:
            alias_key = f"alias#{request.query}"
            response_alias = table.get_item(Key={"canonical_key": alias_key})
            if "Item" in response_alias:
                canonical_key = response_alias["Item"].get("maps_to")
                if canonical_key:
                    logger.info(
                        f"Found alias mapping: {request.query} -> {canonical_key}",
                    )
                    metrics.add_metric(
                        name="AliasMatchFound",
                        unit=MetricUnit.Count,
                        value=1,
                    )

                    canonical_response = table.get_item(
                        Key={"canonical_key": canonical_key},
                    )
                    metadata = (
                        canonical_response.get("Item", {}).get("metadata", {})
                        if "Item" in canonical_response
                        else {}
                    )

                    response = ResolveResponseExact(
                        canonical_key=canonical_key,
                        metadata=metadata,
                        is_alias=True,
                        original_query=request.query,
                    )
                    return response.model_dump()
        except ClientError as e:
            logger.error(f"Error querying alias in DynamoDB: {e}")

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


@app.post("/aliases")
@tracer.capture_method
def create_alias(
    request: CreateAliasRequest,
) -> dict[str, Any] | tuple[dict[str, Any], HTTPStatus]:
    """Create an alias mapping to a canonical tag."""
    logger.info(f"Creating alias: {request.alias} -> {request.canonical_key}")

    try:
        table = get_dynamodb_table()

        canonical_response = table.get_item(
            Key={"canonical_key": request.canonical_key},
        )
        if "Item" not in canonical_response:
            logger.warning(f"Canonical key not found: {request.canonical_key}")
            metrics.add_metric(
                name="AliasCreationFailed",
                unit=MetricUnit.Count,
                value=1,
            )
            return {
                "error": "Canonical key not found",
                "details": (
                    f"The canonical key '{request.canonical_key}' does not "
                    "exist. Please create it first."
                ),
            }, HTTPStatus.NOT_FOUND

        alias_check = table.get_item(Key={"canonical_key": request.alias})
        if "Item" in alias_check:
            logger.warning(f"Alias {request.alias} already exists as canonical key")
            metrics.add_metric(
                name="AliasCreationFailed",
                unit=MetricUnit.Count,
                value=1,
            )
            return {
                "error": "Alias conflict",
                "details": (
                    f"'{request.alias}' already exists as a canonical key. "
                    "Cannot create as alias."
                ),
            }, HTTPStatus.CONFLICT

        alias_key = f"alias#{request.alias}"
        table.put_item(
            Item={
                "canonical_key": alias_key,
                "maps_to": request.canonical_key,
                "metadata": {
                    "type": "alias",
                    "created_at": str(
                        boto3.Session()
                        .client("sts")
                        .get_caller_identity()
                        .get("Account"),
                    ),
                },
            },
        )

        logger.info(
            f"Successfully created alias: {request.alias} -> {request.canonical_key}",
        )
        metrics.add_metric(name="AliasCreated", unit=MetricUnit.Count, value=1)

        response = CreateAliasResponse(
            message=f"Alias '{request.alias}' successfully mapped to '{request.canonical_key}'",
            alias=request.alias,
            canonical_key=request.canonical_key,
        )
        return response.model_dump()

    except Exception:
        logger.exception("Error creating alias")
        metrics.add_metric(name="AliasCreationErrors", unit=MetricUnit.Count, value=1)
        raise


@app.post("/aliases/bulk")
@tracer.capture_method
def create_aliases_bulk(request: CreateAliasesRequest) -> dict[str, Any]:
    """Create multiple alias mappings in bulk."""
    logger.info(f"Creating {len(request.aliases)} aliases in bulk")

    try:
        table = get_dynamodb_table()
        created_count = 0
        failed_count = 0
        errors = []

        with table.batch_writer() as batch:
            for idx, alias_map in enumerate(request.aliases):
                try:
                    alias = alias_map["alias"].strip().lower()
                    canonical_key = alias_map["canonical_key"].strip().lower()

                    # Verify canonical tag exists
                    canonical_response = table.get_item(
                        Key={"canonical_key": canonical_key},
                    )
                    if "Item" not in canonical_response:
                        errors.append(
                            f"Index {idx}: Canonical key '{canonical_key}' not found",
                        )
                        failed_count += 1
                        continue

                    # Create alias entry
                    alias_key = f"alias#{alias}"
                    batch.put_item(
                        Item={
                            "canonical_key": alias_key,
                            "maps_to": canonical_key,
                            "metadata": {"type": "alias"},
                        },
                    )
                    created_count += 1

                except Exception as e:
                    errors.append(f"Index {idx}: {e!s}")
                    failed_count += 1

        logger.info(
            f"Bulk alias creation complete. Created: {created_count}, Failed: {failed_count}",
        )
        metrics.add_metric(
            name="BulkAliasesCreated",
            unit=MetricUnit.Count,
            value=created_count,
        )
        metrics.add_metric(
            name="BulkAliasesFailed",
            unit=MetricUnit.Count,
            value=failed_count,
        )

        response = CreateAliasesResponse(
            message=f"Created {created_count} aliases, {failed_count} failed",
            created_count=created_count,
            failed_count=failed_count,
            errors=errors,
        )
        return response.model_dump()

    except Exception:
        logger.exception("Error creating aliases in bulk")
        metrics.add_metric(
            name="BulkAliasCreationErrors",
            unit=MetricUnit.Count,
            value=1,
        )
        raise


@app.delete("/aliases")
@tracer.capture_method
def delete_alias(
    request: DeleteAliasRequest,
) -> dict[str, Any] | tuple[dict[str, Any], HTTPStatus]:
    """Delete an alias mapping."""
    logger.info(f"Deleting alias: {request.alias}")

    try:
        table = get_dynamodb_table()
        alias_key = f"alias#{request.alias}"

        # Check if alias exists
        alias_response = table.get_item(Key={"canonical_key": alias_key})
        if "Item" not in alias_response:
            logger.warning(f"Alias not found: {request.alias}")
            metrics.add_metric(
                name="AliasDeletionFailed",
                unit=MetricUnit.Count,
                value=1,
            )
            return {
                "error": "Alias not found",
                "details": f"The alias '{request.alias}' does not exist.",
            }, HTTPStatus.NOT_FOUND

        # Delete the alias
        table.delete_item(Key={"canonical_key": alias_key})

        logger.info(f"Successfully deleted alias: {request.alias}")
        metrics.add_metric(name="AliasDeleted", unit=MetricUnit.Count, value=1)

        response = DeleteAliasResponse(
            message=f"Alias '{request.alias}' successfully deleted",
            alias=request.alias,
        )
        return response.model_dump()

    except Exception:
        logger.exception("Error deleting alias")
        metrics.add_metric(name="AliasDeletionErrors", unit=MetricUnit.Count, value=1)
        raise


@app.get("/aliases")
@tracer.capture_method
def list_aliases() -> dict[str, Any]:
    """List all alias mappings."""
    logger.info("Listing all aliases")

    try:
        table = get_dynamodb_table()

        response = table.scan(
            FilterExpression="begins_with(canonical_key, :prefix)",
            ExpressionAttributeValues={":prefix": "alias#"},
        )

        aliases = []
        for item in response.get("Items", []):
            alias_name = item["canonical_key"].replace("alias#", "", 1)
            aliases.append(
                {
                    "alias": alias_name,
                    "canonical_key": item.get("maps_to", ""),
                },
            )

        logger.info(f"Found {len(aliases)} aliases")
        metrics.add_metric(name="AliasesListed", unit=MetricUnit.Count, value=1)

        response = ListAliasesResponse(
            aliases=aliases,
            count=len(aliases),
        )
        return response.model_dump()

    except Exception:
        logger.exception("Error listing aliases")
        metrics.add_metric(name="AliasListingErrors", unit=MetricUnit.Count, value=1)
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
