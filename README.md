# Canonical Tag Resolver

## Problem Statement

In modern editorial newsrooms, editors and journalists need the flexibility to tag content with topics that describe emerging news narratives in real-time. Free-text tagging enables this agility—allowing editorial teams to create new topic tags instantly without bureaucratic approval processes or rigid taxonomies.

However, this freedom comes with a significant challenge: **tag proliferation and duplication**.

### The Challenge

When multiple editors independently create tags for the same underlying concept, you end up with fragmented metadata:

- `Climate Change`, `climate change`, `Climate crisis`, `Global Warming`
- `Artificial Intelligence`, `AI`, `Machine Learning`, `artificial intelligence`
- `United Kingdom`, `UK`, `Britain`, `Great Britain`

This duplication leads to:
- **Inconsistent content organization** - Related articles are scattered across different tags
- **Poor search and discovery** - Users can't find all relevant content in one place
- **Analytics fragmentation** - Editorial insights are split across duplicate tags
- **SEO inefficiency** - Search engines see fragmented topic authority

### The Solution

The Canonical Tag Resolver provides a **hybrid approach** that preserves editorial freedom while ensuring consistency:

1. **Free-text tag creation** - Editors can still create any tag they need, instantly
2. **Intelligent resolution** - The system identifies when a new tag matches an existing canonical tag
3. **Alias management** - Duplicate tags become aliases that automatically resolve to the canonical version
4. **Semantic matching** - Uses vector embeddings and LLM review to detect semantic duplicates, not just exact string matches

This approach gives you the best of both worlds: editorial agility with systematic organization.

---

## Architecture

This system is built on AWS serverless architecture with three core components:

### 1. REST API (AWS Lambda + API Gateway)
Provides endpoints for:
- **Tag resolution** - Check if a tag exists or get suggestions for canonical alternatives
- **Tag ingestion** - Add new canonical tags to the system
- **Alias management** - Create, list, and delete tag aliases
- **Semantic search** - Find related canonical tags using vector similarity

### 2. Vector Search Engine (Meilisearch)
- Stores embeddings of all canonical tags
- Enables semantic similarity search using Azure OpenAI embeddings
- Provides fast, typo-tolerant search capabilities

### 3. Stream Processor (DynamoDB Streams + Lambda)
- Automatically syncs canonical tags to the vector store
- Processes INSERT, UPDATE, and DELETE events from DynamoDB
- Ensures the vector store stays in sync with the canonical tag database

### Data Flow

```
┌─────────────┐
│   Editor    │
│  Creates    │
│  Free Tag   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  /resolve API                           │
│  • Checks for exact match               │
│  • Checks if tag is an alias            │
│  • Performs semantic search             │
│  • Optional: LLM review for auto-merge  │
└──────┬──────────────────────────────────┘
       │
       ▼
┌──────────────────┐        ┌──────────────────┐
│  Exact Match?    │───Yes──│ Return canonical │
│                  │        │ tag immediately  │
└────┬─────────────┘        └──────────────────┘
     │ No
     ▼
┌──────────────────┐        ┌──────────────────┐
│ Semantic Search  │───────▶│ Return similar   │
│ (Vector Store)   │        │ tag suggestions  │
└──────────────────┘        └──────────────────┘
     │
     ▼ (Optional)
┌──────────────────┐        ┌──────────────────┐
│  LLM Review      │───────▶│ Auto-merge or    │
│  (GPT-4)         │        │ require human    │
└──────────────────┘        └──────────────────┘
```

---

## Key Features

### 🎯 Intelligent Tag Resolution
```python
POST /resolve
{
  "query": "climate crisis",
  "similarity_threshold": 0.8,
  "k": 5,
  "enable_llm_review": true
}
```

Returns either:
- **Exact match** - Tag already exists as canonical or alias
- **Suggestions** - Semantically similar canonical tags with similarity scores
- **Auto-merge recommendation** - LLM-reviewed determination if tags can be automatically merged

### 🔗 Alias Management
Create aliases that automatically resolve to canonical tags:

```python
POST /aliases
{
  "alias": "AI",
  "canonical_key": "artificial intelligence"
}
```

Now all references to "AI" automatically resolve to "artificial intelligence".

### 📊 Semantic Search
Find related tags using natural language:

```python
POST /search
{
  "query": "environmental policies",
  "k": 10
}
```

Returns semantically similar canonical tags ranked by similarity score.

### 🤖 LLM-Assisted Review
Optional GPT-4 integration reviews whether tags are truly duplicates:

- Considers context and nuance
- Distinguishes between genuinely different concepts (e.g., "climate change" vs "weather")
- Provides confidence scores and reasoning
- Recommends auto-merge vs. human review

---

## API Endpoints

| Endpoint        | Method | Purpose                                                     |
| --------------- | ------ | ----------------------------------------------------------- |
| `/resolve`      | POST   | Resolve a free-text tag to canonical tag or get suggestions |
| `/search`       | POST   | Semantic search for related canonical tags                  |
| `/ingest`       | POST   | Add new canonical tags to the system                        |
| `/aliases`      | POST   | Create a new alias for a canonical tag                      |
| `/aliases/bulk` | POST   | Create multiple aliases in one request                      |
| `/aliases`      | DELETE | Remove an alias                                             |
| `/aliases`      | GET    | List all aliases for a canonical tag                        |
| `/swagger`      | GET    | API documentation and interactive testing                   |

---

## Technology Stack

- **Runtime**: Python 3.12
- **Cloud Platform**: AWS (Lambda, API Gateway, DynamoDB, DynamoDB Streams)
- **Vector Search**: Meilisearch
- **Embeddings**: Azure OpenAI (text-embedding-3-large)
- **LLM**: Azure OpenAI (GPT-4)
- **Framework**: AWS Powertools for Lambda
- **API Framework**: AWS Lambda Powertools API Gateway Resolver
- **Validation**: Pydantic
- **Infrastructure**: AWS SAM (Serverless Application Model)

---

## Getting Started

### Prerequisites

- AWS Account with appropriate permissions
- Azure OpenAI API access
- Meilisearch instance (hosted or self-hosted)
- Python 3.12+
- AWS SAM CLI
- Docker

### Configuration

Set the following parameters in your SAM deployment:

```yaml
Parameters:
  AzureOpenAIApiKey: <your-azure-openai-api-key>
  MeiliMasterKey: <your-meilisearch-master-key>
  MeiliUrl: <your-meilisearch-url>
  IndexName: semantic_topic
```

### Deployment

```bash
# Build the application
sam build

# Deploy to AWS
sam deploy --guided

# Or use sync for rapid development
sam sync --watch
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Local API testing
sam local start-api
```

---

## Example Usage

### Scenario: Editor Creates New Tag

An editor wants to tag an article with "climate crisis":

1. **Editor submits tag**: `climate crisis`
2. **System resolves**:
   ```python
   POST /resolve
   {"query": "climate crisis", "enable_llm_review": true}
   ```
3. **System responds**:
   ```json
   {
     "found": false,
     "match_type": "suggestions",
     "suggestions": [
       {
         "canonical_key": "climate change",
         "similarity_score": 0.92,
         "llm_review": {
           "can_auto_merge": true,
           "confidence": 0.95,
           "reasoning": "Both terms refer to the same phenomenon..."
         }
       }
     ]
   }
   ```
4. **Editor decision**:
   - Accept suggestion → Create alias: `climate crisis` → `climate change`
   - Create new canonical tag → Add `climate crisis` as new canonical tag
   - Override → Use free-text tag without aliasing

---

## Benefits

### For Editorial Teams
- ✅ Maintain creative freedom and agility
- ✅ Reduce cognitive load (no need to remember exact tag names)
- ✅ Get instant suggestions for existing tags
- ✅ Prevent accidental duplication

### For Content Organization
- ✅ Consistent taxonomy without rigid constraints
- ✅ Automatic resolution of duplicate concepts
- ✅ Better content discovery and navigation
- ✅ Improved analytics and insights

### For Technical Teams
- ✅ Serverless architecture (auto-scaling, pay-per-use)
- ✅ Event-driven synchronization (DynamoDB Streams)
- ✅ Observable with AWS Powertools (logging, metrics, tracing)
- ✅ Type-safe with Pydantic validation
- ✅ OpenAPI/Swagger documentation

---

## License

This is a proof-of-concept project. License terms to be determined.

---

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## Contact

For questions or feedback, please open an issue in this repository.
