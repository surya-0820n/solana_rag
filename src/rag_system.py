from typing import List, Dict, Any, Tuple, Optional
from .text_processor import DiscordTextProcessor, TwitterProcessor
import os
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import httpx

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with text processors for different data sources"""
        self.text_processor = DiscordTextProcessor()
        self.twitter_processor = TwitterProcessor()
        self.use_openai = bool(os.getenv('OPENAI_API_KEY'))
        self.openai_client = None
        if self.use_openai:
            try:
                # Create a custom HTTP client without proxy settings
                http_client = httpx.Client()
                self.openai_client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    http_client=http_client
                )
                logger.info("Successfully initialized OpenAI client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise
        logger.info(f"RAG System initialized with OpenAI support: {self.use_openai}")
        
        # Load sentence-transformers model
        self.model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
        
        logger.info("RAGSystem initialized with sentence-transformers")
        
    def process_and_store_messages(self, messages: List[Dict[str, Any]]):
        """Process and store messages in Pinecone"""
        if not messages:
            return
            
        # Process messages
        processed_messages = [
            self.text_processor.process_message(message)
            for message in messages
        ]
        
        # Store in Pinecone
        self.text_processor.upsert_to_pinecone(processed_messages)
        
    def generate_response(self, query: str, model: str = "auto") -> str:
        """Generate a response to a query using the RAG system"""
        # Search for similar messages
        discord_matches = self.text_processor.search_similar_messages(query)
        tweet_matches = self.twitter_processor.search_similar_tweets(query)
        
        # Combine and format context
        context = self._format_context(discord_matches, tweet_matches)
        
        # Generate response using the appropriate model
        if model == "auto":
            # Choose model based on context length
            if len(context) > 2000:
                model = "openai"
            else:
                model = "sentence-transformers"
                
        if model == "openai":
            return self._generate_openai_response(query, context)
        else:
            return self._generate_local_response(query, context)
            
    def generate_response_with_context(
        self, 
        query: str, 
        model: str = "auto",
        top_k: int = 3,
        data_source_priority: str = "both"
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate a response with relevant context from both data sources"""
        # Get matches based on priority
        discord_matches = []
        tweet_matches = []
        
        if data_source_priority in ["both", "discord"]:
            discord_matches = self.text_processor.search_similar_messages(query, top_k)
            
        if data_source_priority in ["both", "twitter"]:
            tweet_matches = self.twitter_processor.search_similar_tweets(query, top_k)
            
        # Generate response
        response = self.generate_response(query, model)
        
        return response, discord_matches, tweet_matches
        
    def _format_context(self, discord_matches: List[Dict[str, Any]], tweet_matches: List[Dict[str, Any]]) -> str:
        """Format matches into a context string"""
        context_parts = []
        
        # Add Discord messages
        if discord_matches:
            context_parts.append("Relevant Discord messages:")
            for match in discord_matches:
                context_parts.append(f"- {match.metadata['text']} (by {match.metadata['author']})")
                
        # Add tweets
        if tweet_matches:
            context_parts.append("\nRelevant tweets:")
            for match in tweet_matches:
                context_parts.append(f"- {match.metadata['text']} (by {match.metadata['author']})")
                
        return "\n".join(context_parts)
        
    def _generate_openai_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI's API"""
        try:
            additional_context = """
            Also refer to the following websites depending on the question:
            
            Solana Delegation criteria: https://solana.org/delegation-criteria
            
            """
            logger.info(f"Generating response using OpenAI")
            prompt = f"""
                    You are a helpful assistant that answers questions about Solana blockchain.

                    Context:
                    {context}

                    {additional_context}

                    Question:
                    {query}
                    """
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            logger.info("Successfully generated response using OpenAI")
            return answer
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise
        
    def _generate_local_response(self, query: str, context: str) -> str:
        """Generate response using local sentence-transformers model"""
        try:
            logger.info("Generating response using sentence-transformers")
            # Combine context and question
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            
            # Generate response using the model
            # Note: This is a simple implementation. You might want to use a more sophisticated approach
            # like using a smaller language model or a different strategy for sentence-transformers
            answer = "Based on the provided context:\n\n"
            answer += context  # For now, we'll return the context as the answer
            answer += "\n\nPlease refer to the above context for the answer to your question."
            
            return answer
        except Exception as e:
            logger.error(f"Error generating sentence-transformers response: {str(e)}")
            raise 