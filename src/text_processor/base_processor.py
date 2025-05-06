from typing import List, Dict, Any, Optional
from loguru import logger
import os
from dotenv import load_dotenv
from datetime import datetime
from ..utils.singletons import get_model, get_index

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
        
        # Get model and index from singletons
        self.model = get_model()
        self.index = get_index(index_name, dimension)
        
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        return self.model.encode(texts).tolist()
    
    def check_index_stats(self) -> Dict[str, Any]:
        """Check Pinecone index statistics"""
        return self.index.describe_index_stats()
    
    def create_index(self, dimension: Optional[int] = None):
        """Create a new Pinecone index"""
        if dimension is None:
            dimension = self.dimension
            
        # This will create the index if it doesn't exist
        self.index = get_index(self.index_name, dimension)
    
    def delete_vectors_by_date_range(self, start_date: str, end_date: str) -> int:
        """Delete vectors from Pinecone based on their creation date range"""
        # Convert dates to timestamps
        start_timestamp = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
        end_timestamp = datetime.strptime(end_date, "%Y-%m-%d").timestamp()
        
        # Get all vectors in the date range
        vectors = self.index.fetch(ids=[])
        deleted_count = 0
        
        for vector_id, vector in vectors.items():
            if start_timestamp <= vector.metadata.get('timestamp', 0) <= end_timestamp:
                self.index.delete(ids=[vector_id])
                deleted_count += 1
                
        return deleted_count 