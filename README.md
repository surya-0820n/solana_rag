# Solana RAG System

A Retrieval Augmented Generation (RAG) system for answering Solana-related questions using Discord community discussions.

## Features

- Fetches messages from a specified Discord channel
- Stores messages in PostgreSQL database
- Processes and embeds messages using sentence-transformers
- Stores embeddings in Pinecone vector database
- Provides a REST API for asking questions
- Uses GPT-4 to generate answers based on retrieved context

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Discord Bot Token
- OpenAI API Key
- Pinecone API Key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd solana_rag
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the environment template and fill in your credentials:
```bash
cp .env.example .env
```

5. Set up the database:
```bash
# Create PostgreSQL database
createdb solana_rag

# Run database migrations (if using Alembic)
alembic upgrade head
```

6. Create a Pinecone index:
- Go to Pinecone console
- Create a new index with dimension 384 (for all-MiniLM-L6-v2 model)
- Copy the index name to your .env file

## Usage

1. Start the API server:
```bash
python -m src.api
```

2. Fetch messages from Discord:
```bash
curl -X POST http://localhost:8000/fetch-messages
```

3. Ask questions:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I create a token account on Solana?"}'
```

## API Endpoints

- `POST /ask`: Ask a question about Solana
- `POST /fetch-messages`: Fetch new messages from Discord and update the knowledge base

## Architecture

1. **Data Collection**:
   - Discord bot fetches messages from specified channel
   - Messages are stored in PostgreSQL database

2. **Text Processing**:
   - Messages are cleaned and chunked
   - Chunks are embedded using sentence-transformers
   - Embeddings are stored in Pinecone

3. **Question Answering**:
   - User question is embedded
   - Similar chunks are retrieved from Pinecone
   - Context is combined with question
   - GPT-4 generates answer based on context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 