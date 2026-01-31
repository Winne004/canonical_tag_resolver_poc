import json
from functools import lru_cache
from typing import Any

from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.data_classes import (
    DynamoDBStreamEvent,
    event_source,
)
from aws_lambda_powertools.utilities.data_classes.dynamo_db_stream_event import (
    DynamoDBRecordEventName,
)
from aws_lambda_powertools.utilities.typing import LambdaContext
from langchain_community.vectorstores import Meilisearch
from langchain_openai import AzureOpenAIEmbeddings
from meilisearch import Client as MeiliClient

from settings import AzureOpenAISettings, MeilisearchSettings

# Initialize Powertools
logger = Logger()
tracer = Tracer()
metrics = Metrics()


@lru_cache(maxsize=1)
def get_azure_settings() -> AzureOpenAISettings:
    """Get or create Azure OpenAI settings (cached)."""
    return AzureOpenAISettings()  # pyright: ignore[reportCallIssue]


@lru_cache(maxsize=1)
def get_meili_settings() -> MeilisearchSettings:
    """Get or create Meilisearch settings (cached)."""
    return MeilisearchSettings()  # pyright: ignore[reportCallIssue]


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


@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
@event_source(data_class=DynamoDBStreamEvent)
def lambda_handler(
    event: DynamoDBStreamEvent, context: LambdaContext
) -> dict[str, Any]:
    """Lambda handler for processing DynamoDB stream events to sync with vector store."""
    records = list(event.records)
    logger.info("Processing %d DynamoDB stream records", len(records))

    azure_settings = get_azure_settings()
    meili_settings = get_meili_settings()
    meili_client = get_meili_client()
    embeddings = get_embeddings()

    processed = 0
    errors = 0

    for record in records:
        try:
            event_name = record.event_name
            logger.info(f"Processing event: {event_name}")

            if event_name == DynamoDBRecordEventName.INSERT:
                if not record.dynamodb:
                    logger.warning("Skipping record with no dynamodb data")
                    continue
                new_image = record.dynamodb.new_image
                canonical_key = new_image["canonical_key"]

                logger.info(f"Adding canonical key to vector store: {canonical_key}")

                embedders = {
                    azure_settings.embedding_model: {
                        "source": "userProvided",
                        "dimensions": azure_settings.dimensions,
                    },
                }

                Meilisearch.from_texts(
                    client=meili_client,
                    index_name=meili_settings.index_name,
                    texts=[canonical_key],
                    embedding=embeddings,
                    embedders=embedders,
                    embedder_name=azure_settings.embedding_model,
                )

                metrics.add_metric(
                    name="VectorStoreInsert",
                    unit=MetricUnit.Count,
                    value=1,
                )
                processed += 1

            elif event_name == DynamoDBRecordEventName.MODIFY:
                if not record.dynamodb:
                    logger.warning("Skipping record with no dynamodb data")
                    continue
                old_image = record.dynamodb.old_image
                new_image = record.dynamodb.new_image
                old_key = old_image["canonical_key"]
                new_key = new_image["canonical_key"]

                if old_key != new_key:
                    logger.info(f"Updating canonical key: {old_key} -> {new_key}")

                    index = meili_client.index(meili_settings.index_name)
                    search_results = index.search(old_key, limit=1)
                    if search_results.get("hits"):
                        doc_id = search_results["hits"][0]["id"]
                        index.delete_document(doc_id)
                        logger.info(f"Deleted old key from vector store: {old_key}")

                    embedders = {
                        azure_settings.embedding_model: {
                            "source": "userProvided",
                            "dimensions": azure_settings.dimensions,
                        },
                    }

                    Meilisearch.from_texts(
                        client=meili_client,
                        index_name=meili_settings.index_name,
                        texts=[new_key],
                        embedding=embeddings,
                        embedders=embedders,
                        embedder_name=azure_settings.embedding_model,
                    )

                    metrics.add_metric(
                        name="VectorStoreUpdate",
                        unit=MetricUnit.Count,
                        value=1,
                    )
                    processed += 1
                else:
                    logger.info(
                        f"No key change detected, skipping update for: {old_key}",
                    )

            elif event_name == DynamoDBRecordEventName.REMOVE:
                if not record.dynamodb:
                    logger.warning("Skipping record with no dynamodb data")
                    continue
                old_image = record.dynamodb.old_image
                canonical_key = old_image["canonical_key"]

                logger.info(
                    f"Removing canonical key from vector store: {canonical_key}",
                )

                index = meili_client.index(meili_settings.index_name)
                search_results = index.search(canonical_key, limit=1)
                if search_results.get("hits"):
                    doc_id = search_results["hits"][0]["id"]
                    index.delete_document(doc_id)
                    logger.info(f"Deleted key from vector store: {canonical_key}")
                    metrics.add_metric(
                        name="VectorStoreDelete",
                        unit=MetricUnit.Count,
                        value=1,
                    )
                    processed += 1
                else:
                    logger.warning(f"Key not found in vector store: {canonical_key}")

        except Exception as e:
            logger.exception(f"Error processing stream record: {e}")
            metrics.add_metric(
                name="StreamProcessingErrors",
                unit=MetricUnit.Count,
                value=1,
            )
            errors += 1

    logger.info(f"Stream processing complete. Processed: {processed}, Errors: {errors}")
    metrics.add_metric(
        name="StreamRecordsProcessed",
        unit=MetricUnit.Count,
        value=processed,
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "processed": processed,
                "errors": errors,
            },
        ),
    }
