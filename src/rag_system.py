from typing import List, Dict, Any, Tuple, Optional
from .text_processor.discord_processor import DiscordTextProcessor
from .text_processor.twitter_processor import TwitterProcessor
import os
from tqdm import tqdm
from loguru import logger
from .utils.singletons import get_model
from openai import OpenAI
import numpy as np

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with text processors for different data sources"""
        self.text_processor = DiscordTextProcessor()
        self.twitter_processor = TwitterProcessor()
        self.use_openai = bool(os.getenv('OPENAI_API_KEY'))
        self.openai_client = None
        if self.use_openai:
            try:
                # Initialize OpenAI client with API key
                self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                logger.info("Successfully initialized OpenAI client")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                raise
        logger.info(f"RAG System initialized with OpenAI support: {self.use_openai}")
        
        # Get model from singleton
        self.model = get_model()
        
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
            context_parts.append("Discord Messages:")
            for match in discord_matches:
                context_parts.append(f"- {match['metadata']['text']}")
                
        # Add Twitter messages
        if tweet_matches:
            context_parts.append("\nTwitter Messages:")
            for match in tweet_matches:
                context_parts.append(f"- {match['metadata']['text']}")
                
        return "\n".join(context_parts)
        
    def _generate_openai_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI's API"""
        try:
            additional_context = """
            Also refer to the following websites depending on the question:
            
            Solana Delegation criteria: https://solana.org/delegation-criteria
            Solana Hardware Compatibility: https://solanahcl.org/
            Solana Official Documentation: https://solana.com/docs
            Agave Validator Documentation: https://docs.anza.xyz/
            """
            logger.info(f"Generating response using OpenAI")
            prompt = f"""
                    You are a knowledgeable and helpful assistant specializing in Solana blockchain. 
                    Your goal is to provide clear, accurate, and helpful answers to questions about Solana.
                    
                    You have access to the following information:
                    {context}
    
                    {additional_context}
                    
                    GUIDELINES for your response:
                    1. Answer naturally and conversationally, as if you're explaining to a friend
                    2. Don't directly quote or reference the context - use it to inform your answer
                    3. If you're not sure about something, say so
                    4. Keep your answers clear and concise
                    5. If relevant, you can use information from the websites in the context
                    6. If the question is very nuanced, you deep dive and use the websites to provide more information.
                    7. If you find different information from different sources, stick to the latest information and provide a link to the source.
                    

                    Give CLEAR and CONCISE answer, don't be verbose. Your answer should NOT be big paragraph, but bullet points and MUST have relevant metrics
                    Question:
                    {query}
                    """
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
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
        # Create embeddings for query and context
        query_embedding = self.model.encode(query)
        context_embedding = self.model.encode(context)
        
        # Calculate similarity
        similarity = np.dot(query_embedding, context_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding))
        
        if similarity < 0.5:
            return "I don't have enough relevant information to answer this question."
            
        # For local model, we'll just return the most relevant context
        return f"Based on the available information: {context}" 