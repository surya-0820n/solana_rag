import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

class TextProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        pinecone.init(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT')
        )
        self.index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
        
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
        vectors = []
        for chunk in chunks:
            vectors.append({
                'id': f"{chunk['metadata']['message_id']}_{chunk['metadata']['chunk_index']}",
                'values': chunk['embedding'],
                'metadata': chunk['metadata']
            })
        
        self.index.upsert(vectors=vectors)
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks in the vector database"""
        query_embedding = self.model.encode(query)
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        return results.matches 