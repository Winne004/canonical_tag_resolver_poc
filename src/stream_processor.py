import json
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

from src.shared import (
    get_azure_settings,
    get_meili_client,
    get_meili_settings,
    get_vector_store,
)

# Initialize Powertools
logger = Logger()
tracer = Tracer()
metrics = Metrics()


def add_to_vector_store(canonical_key: str) -> None:
    """Add a canonical key to the vector store.

    Uses cached vector store to avoid updating index settings on every call.
    """
    azure_settings = get_azure_settings()
    vector_store = get_vector_store()

    # Use add_texts instead of from_texts to avoid recreating the vector store
    # and calling update_embedders() on every document add
    vector_store.add_texts(
        texts=[canonical_key],
        embedder_name=azure_settings.embedding_model,
    )


def delete_from_vector_store(canonical_key: str) -> bool:
    """Delete a canonical key from the vector store.

    Returns:
        True if deleted successfully, False if not found.

    """
    meili_settings = get_meili_settings()
    meili_client = get_meili_client()
    index = meili_client.index(meili_settings.index_name)

    search_results = index.search(canonical_key, limit=1)
    if search_results.get("hits"):
        doc_id = search_results["hits"][0]["id"]
        index.delete_document(doc_id)
        logger.info(f"Deleted key from vector store: {canonical_key}")
        return True
    return False


@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
@event_source(data_class=DynamoDBStreamEvent)
def lambda_handler(
    event: DynamoDBStreamEvent,
    context: LambdaContext,
) -> dict[str, Any]:
    """Lambda handler for processing DynamoDB stream events to sync with vector store."""
    records = list(event.records)
    logger.info("Processing %d DynamoDB stream records", len(records))

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
                add_to_vector_store(canonical_key)
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
                    delete_from_vector_store(old_key)
                    add_to_vector_store(new_key)
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
                if delete_from_vector_store(canonical_key):
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
