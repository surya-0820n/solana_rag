import pinecone
from sentence_transformers import SentenceTransformer
from loguru import logger
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from datetime import datetime

load_dotenv()

class BaseTextProcessor:
    def __init__(self, index_name: str, dimension: int = 384):
        """Initialize the base text processor
        
        Args:
            index_name (str): Name of the Pinecone index
            dimension (int, optional): Dimension of the embeddings. Defaults to 384.
        """
        self.index_name = index_name
        self.dimension = dimension
        
        # Initialize Pinecone
        logger.info("Initializing Pinecone client")
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT')
        )
        logger.info("Successfully initialized Pinecone client")
        
        # Load sentence transformer model
        logger.info("Loading sentence-transformers model")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded sentence-transformers model")
        
        # Create or connect to Pinecone index
        self._setup_index()
        
    def _setup_index(self):
        """Setup the Pinecone index"""
        if self.index_name not in pinecone.list_indexes():
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine"
            )
        else:
            logger.info(f"Using existing index: {self.index_name}")
            
        self.pc = pinecone.Index(self.index_name)
        logger.info("Successfully connected to Pinecone index")
        
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        return self.model.encode(texts).tolist()
    
    def check_index_stats(self) -> Dict[str, Any]:
        """Check Pinecone index statistics"""
        return self.pc.describe_index_stats()
    
    def create_index(self, dimension: Optional[int] = None):
        """Create a new Pinecone index"""
        if dimension is None:
            dimension = self.dimension
            
        pinecone.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine"
        )
        self.pc = pinecone.Index(self.index_name)
        
    def delete_vectors_by_date_range(self, start_date: str, end_date: str) -> int:
        """Delete vectors from Pinecone based on their creation date range"""
        # Convert dates to timestamps
        start_timestamp = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
        end_timestamp = datetime.strptime(end_date, "%Y-%m-%d").timestamp()
        
        # Get all vectors in the date range
        vectors = self.pc.fetch(ids=[])
        deleted_count = 0
        
        for vector_id, vector in vectors.items():
            if start_timestamp <= vector.metadata.get('timestamp', 0) <= end_timestamp:
                self.pc.delete(ids=[vector_id])
                deleted_count += 1
                
        return deleted_count 