from pydantic import SecretStr
from pydantic_settings import BaseSettings


class AzureOpenAISettings(BaseSettings):
    """Configuration settings for OpenAI API."""

    azure_open_ai_api_key: SecretStr
    llm_model: str = "gpt-4.1-nano"
    llm_deployment: str = "gpt-4.1-nano"
    llm_api_version: str = "2024-12-01-preview"
    endpoint: str = "https://uksouth.api.cognitive.microsoft.com/"
    embedding_model: str = "text-embedding-3-large"
    dimensions: int = 3072


class MeilisearchSettings(BaseSettings):
    """Configuration settings for Meilisearch."""

    meli_master_key: SecretStr
    meili_url: str = "https://edge.meilisearch.com"
    index_name: str = "semantic_topic"


class DynamoDBSettings(BaseSettings):
    """Configuration settings for DynamoDB."""

    canonical_tags_table: str
