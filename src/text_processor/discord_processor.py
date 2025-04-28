from typing import List, Dict, Any
from loguru import logger
from .base_processor import BaseTextProcessor
import os

class DiscordTextProcessor(BaseTextProcessor):
    def __init__(self):
        """Initialize the Discord text processor"""
        super().__init__(
            index_name=os.getenv('PINECONE_INDEX_NAME', 'solana-rag'),
            dimension=384
        )
        logger.info("DiscordTextProcessor initialized successfully")
        
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single Discord message for storage in Pinecone"""
        # Create embedding for the message content
        embedding = self.create_embeddings([message['content']])[0]
        
        # Prepare metadata
        metadata = {
            'text': message['content'],
            'author': message['author']['username'],
            'timestamp': message['timestamp'],
            'channel_id': message['channel_id'],
            'message_id': message['id']
        }
        
        return {
            'id': message['id'],
            'values': embedding,
            'metadata': metadata
        }
        
    def upsert_to_pinecone(self, processed_messages: List[Dict[str, Any]]):
        """Upsert processed messages to Pinecone"""
        if not processed_messages:
            return
            
        # Prepare vectors for upsert
        vectors = [
            {
                'id': msg['id'],
                'values': msg['values'],
                'metadata': msg['metadata']
            }
            for msg in processed_messages
        ]
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.pc.upsert(vectors=batch)
            
        logger.info(f"Successfully upserted {len(vectors)} messages to Pinecone")
        
    def search_similar_messages(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar messages in Pinecone"""
        # Create embedding for the query
        query_embedding = self.create_embeddings([query])[0]
        
        # Search in Pinecone
        results = self.pc.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches 