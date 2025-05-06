import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure required environment variables are set
required_vars = [
    'OPENAI_API_KEY',
    'PINECONE_API_KEY',
    'PINECONE_ENVIRONMENT',
    'PINECONE_INDEX_NAME'
]

for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}") 