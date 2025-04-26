import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from loguru import logger
import time
from pinecone import Pinecone, ServerlessSpec
import uuid

# Load environment variables
load_dotenv()

class TextProcessor:
    def __init__(self):
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'solana-rag')
        
        if not all([self.pinecone_api_key, self.pinecone_env]):
            logger.error("Missing Pinecone configuration. Please check your .env file.")
            logger.error(f"PINECONE_API_KEY: {'Present' if self.pinecone_api_key else 'Missing'}")
            logger.error(f"PINECONE_ENVIRONMENT: {'Present' if self.pinecone_env else 'Missing'}")
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
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully loaded sentence-transformers model")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {str(e)}")
            raise
        
        # Create or connect to Pinecone index
        try:
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinezcone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.pinecone_env
                    )
                )
                logger.info(f"Successfully created index: {self.index_name}")
            else:
                logger.info(f"Using existing index: {self.index_name}")
                
            self.index = self.pc.Index(self.index_name)
            logger.info("Successfully connected to Pinecone index")
        except Exception as e:
            logger.error(f"Failed to create/connect to Pinecone index: {str(e)}")
            raise
        
        logger.info("TextProcessor initialized successfully")

    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Get relevant context from Pinecone for a given query"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format context from results
            context = "Here are some relevant excerpts from Solana community discussions:\n\n"
            
            for i, match in enumerate(results.matches, 1):
                context += f"Excerpt {i}:\n"
                context += f"Content: {match.metadata.get('text', '')}\n"
                context += f"Author: {match.metadata.get('author', 'Unknown')}\n"
                context += f"Date: {match.metadata.get('timestamp', 'Unknown')}\n"
                context += f"Relevance Score: {match.score:.2f}\n\n"
            
            if not results.matches:
                context = "No relevant context found in the knowledge base."
            
            logger.info(f"Retrieved {len(results.matches)} relevant documents")
            from pprint import pprint
            pprint(results.matches)
            return context.strip()
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise

    def process_and_store(self, text: str, metadata: dict) -> None:
        """Process text and store in Pinecone"""
        try:
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Prepare metadata
            metadata['text'] = text
            
            # Store in Pinecone
            self.index.upsert(
                vectors=[{
                    'id': str(uuid.uuid4()),
                    'values': embedding,
                    'metadata': metadata
                }]
            )
            
            logger.info(f"Successfully processed and stored text with metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error processing and storing text: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean the text by removing unnecessary elements"""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove inline code
        text = re.sub(r'`.*?`', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emojis
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
    
    def process_message(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single message into chunks with embeddings"""
        cleaned_text = self.clean_text(message['content'])
        chunks = self.chunk_text(cleaned_text)
        
        # Generate embeddings for chunks
        embeddings = self.model.encode(chunks)
        
        # Prepare chunks with metadata
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                'text': chunk,
                'embedding': embedding.tolist(),
                'metadata': {
                    'message_id': message['id'],
                    'chunk_index': i,
                    'author': message['author']['username'],
                    'timestamp': message['timestamp'].isoformat(),
                    'channel_id': message['channel_id']
                }
            })
        
        return processed_chunks
    
    def upsert_to_pinecone(self, chunks: List[Dict[str, Any]]):
        """Upload chunks to Pinecone vector database"""
        try:
            logger.info(f"Upserting {len(chunks)} chunks to Pinecone")
            vectors = []
            for chunk in chunks:
                # Ensure embedding is a list of floats
                embedding = chunk['embedding']
                if not isinstance(embedding, list):
                    embedding = embedding.tolist()
                
                # Create vector with correct format
                vector = {
                    'id': f"{chunk['metadata']['message_id']}_{chunk['metadata']['chunk_index']}",
                    'values': embedding,
                    'metadata': {
                        'text': chunk['text'],
                        'message_id': chunk['metadata']['message_id'],
                        'chunk_index': chunk['metadata']['chunk_index'],
                        'author': chunk['metadata']['author'],
                        'timestamp': chunk['metadata']['timestamp'],
                        'channel_id': chunk['metadata']['channel_id']
                    }
                }
                vectors.append(vector)
            
            # Upsert in batches to avoid request size limits
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                logger.info(f"Upserting batch {i//batch_size + 1} of {len(vectors)//batch_size + 1}")
                self.index.upsert(vectors=batch)
                
            logger.info("Successfully upserted all vectors to Pinecone")
        except Exception as e:
            logger.error(f"Error upserting to Pinecone: {str(e)}")
            raise e
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks in the vector database"""
        query_embedding = self.model.encode(query)
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results.matches
    
    def check_index_stats(self):
        """Check and display Pinecone index statistics"""
        try:
            # Get index statistics
            stats = self.index.describe_index_stats()
            
            # Get sample vectors
            sample_vectors = self.index.query(
                vector=[0.0] * 384,  # Query with zero vector to get random results
                top_k=5,
                include_metadata=True
            )
            
            logger.info("\n=== Pinecone Index Statistics ===")
            logger.info(f"Index Name: {os.getenv('PINECONE_INDEX_NAME')}")
            logger.info(f"Total Vectors: {stats['total_vector_count']}")
            logger.info(f"Dimension: {stats['dimension']}")
            
            logger.info("\n=== Sample Vectors ===")
            for i, match in enumerate(sample_vectors.matches, 1):
                logger.info(f"\nSample {i}:")
                logger.info(f"ID: {match.id}")
                logger.info(f"Score: {match.score}")
                logger.info(f"Metadata: {match.metadata}")
                
            return stats
        except Exception as e:
            logger.error(f"Error checking index stats: {str(e)}")
            raise e
    
    def create_index(self, dimension: int = 384):
        """Create a new Pinecone index with the correct dimension"""
        try:
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index_name = os.getenv('PINECONE_INDEX_NAME')
            
            # Delete existing index if it exists
            if index_name in pc.list_indexes().names():
                logger.info(f"Deleting existing index {index_name}")
                pc.delete_index(index_name)
            
            # Create new index with correct dimension
            logger.info(f"Creating new index {index_name} with dimension 384")
            pc.create_index(
                name=index_name,
                dimension=dimension,  # Dimension for all-MiniLM-L6-v2 model
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            
            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                logger.info("Waiting for index to be ready...")
                time.sleep(1)
            
            logger.info("Index created successfully")
            self.index = pc.Index(index_name)
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise e 