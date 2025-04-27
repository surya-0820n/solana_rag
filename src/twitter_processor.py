from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from loguru import logger
from pinecone import Pinecone
import uuid
import re

# Load environment variables
load_dotenv()

class TwitterProcessor:
    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
        self.index_name = os.getenv('TWITTER_PINECONE_INDEX_NAME', 'solana-rag-twitter')
        
        if not all([self.pinecone_api_key, self.pinecone_env]):
            logger.error("Missing Pinecone configuration. Please check your .env file.")
            raise ValueError("Missing Pinecone configuration")
            
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            logger.info(f"Successfully initialized Pinecone client")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise
            
        # Initialize sentence-transformers model
        try:
            self.model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
            logger.info("Successfully loaded sentence-transformers model")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {str(e)}")
            raise
            
        # Create or connect to Pinecone index
        try:
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Twitter Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2 model
                    metric="cosine"
                )
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Using existing Twitter index: {self.index_name}")
                
            self.index = self.pc.Index(self.index_name)
            logger.info("TwitterProcessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to create/connect to Pinecone index: {str(e)}")
            raise
            
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
        
    def process_tweet(self, tweet: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single tweet into chunks with embeddings"""
        cleaned_text = self.clean_tweet_text(tweet['text'])
        
        # Generate embedding
        embedding = self.model.encode(cleaned_text).tolist()
        
        # Prepare tweet with metadata
        processed_tweet = {
            'id': str(uuid.uuid4()),
            'values': embedding,
            'metadata': {
                'text': cleaned_text,
                'tweet_id': tweet['id'],
                'author': tweet['author']['username'],
                'created_at': tweet['created_at'].isoformat(),
                'metrics': tweet['metrics'],
                'entities': tweet['entities']
            }
        }
        
        return processed_tweet
        
    def upsert_to_pinecone(self, tweets: List[Dict[str, Any]]):
        """Upload processed tweets to Pinecone vector database"""
        try:
            logger.info(f"Upserting {len(tweets)} tweets to Pinecone")
            
            # Process tweets in batches
            batch_size = 100
            for i in range(0, len(tweets), batch_size):
                batch = tweets[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1} of {len(tweets)//batch_size + 1}")
                
            logger.info("Successfully upserted all tweets to Pinecone")
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {str(e)}")
            raise
            
    def get_relevant_tweets(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant tweets from Pinecone for a given query"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            relevant_tweets = []
            for match in results.matches:
                relevant_tweets.append({
                    'text': match.metadata['text'],
                    'author': match.metadata['author'],
                    'created_at': match.metadata['created_at'],
                    'metrics': match.metadata['metrics'],
                    'score': match.score
                })
            
            logger.info(f"Retrieved {len(relevant_tweets)} relevant tweets")
            return relevant_tweets
            
        except Exception as e:
            logger.error(f"Error retrieving relevant tweets: {str(e)}")
            raise 