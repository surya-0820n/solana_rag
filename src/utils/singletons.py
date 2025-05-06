from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

# Module-level singleton instances
_model = None
_pinecone_client = None
_indices: Dict[str, Any] = {}

def get_model():
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded sentence-transformers model")
    return _model

def get_pinecone_client():
    global _pinecone_client
    if _pinecone_client is None:
        logger.info("Initializing Pinecone client")
        _pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        logger.info("Successfully initialized Pinecone client")
    return _pinecone_client

def get_index(index_name: str, dimension: int = 384):
    global _indices
    if index_name not in _indices:
        pc = get_pinecone_client()
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
            logger.info(f"Successfully created index: {index_name}")
        else:
            logger.info(f"Using existing index: {index_name}")
            
        _indices[index_name] = pc.Index(index_name)
        logger.info("Successfully connected to Pinecone index")
    return _indices[index_name] 