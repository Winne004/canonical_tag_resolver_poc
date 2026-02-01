from functools import lru_cache

import boto3
from langchain_community.vectorstores import Meilisearch
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from meilisearch import Client as MeiliClient

from src.settings import AzureOpenAISettings, DynamoDBSettings, MeilisearchSettings


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


@lru_cache(maxsize=1)
def get_embeddings() -> AzureOpenAIEmbeddings:
    """Get or create Azure OpenAI embeddings (cached)."""
    azure_settings = get_azure_settings()
    return AzureOpenAIEmbeddings(
        model=azure_settings.embedding_model,
        azure_endpoint=azure_settings.endpoint,
        api_key=azure_settings.azure_open_ai_api_key,
    )
