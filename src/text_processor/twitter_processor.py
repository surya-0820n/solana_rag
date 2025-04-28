from typing import List, Dict, Any
from loguru import logger
import os
from .base_processor import BaseTextProcessor
import re

class TwitterProcessor(BaseTextProcessor):
    def __init__(self):
        """Initialize the Twitter text processor"""
        super().__init__(
            index_name=os.getenv('TWITTER_PINECONE_INDEX_NAME', 'solana-rag-twitter'),
            dimension=384
        )
        logger.info("TwitterProcessor initialized successfully")
        
    def process_tweet(self, tweet: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single tweet for storage in Pinecone"""
        # Clean tweet text
        cleaned_text = self.clean_tweet_text(tweet['text'])
        
        # Create embedding for the tweet text
        embedding = self.create_embeddings([cleaned_text])[0]
        
        # Prepare metadata
        metadata = {
            'text': cleaned_text,
            'author': tweet['author']['username'],
            'timestamp': tweet['created_at'],
            'tweet_id': tweet['id'],
            'url': tweet.get('url', '')
        }
        
        return {
            'id': tweet['id'],
            'values': embedding,
            'metadata': metadata
        }
        
    def upsert_to_pinecone(self, processed_tweets: List[Dict[str, Any]]):
        """Upsert processed tweets to Pinecone"""
        if not processed_tweets:
            return
            
        # Prepare vectors for upsert
        vectors = [
            {
                'id': tweet['id'],
                'values': tweet['values'],
                'metadata': tweet['metadata']
            }
            for tweet in processed_tweets
        ]
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.pc.upsert(vectors=batch)
            
        logger.info(f"Successfully upserted {len(vectors)} tweets to Pinecone")
        
    def search_similar_tweets(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar tweets in Pinecone"""
        # Create embedding for the query
        query_embedding = self.create_embeddings([query])[0]
        
        # Search in Pinecone
        results = self.pc.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches

    def clean_tweet_text(self, text: str) -> str:
        """Clean tweet text by removing URLs, mentions, and other noise"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (but keep the text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip() 