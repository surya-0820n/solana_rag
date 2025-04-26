from typing import List, Dict, Any
from .text_processor import TextProcessor
import os
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import httpx

class RAGSystem:
    def __init__(self):
        self.text_processor = TextProcessor()
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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("RAGSystem initialized with sentence-transformers")
        
    def format_context(self, matches: List[Dict[str, Any]]) -> str:
        """Format retrieved context for the language model"""
        context = "Here are some relevant excerpts from Solana community discussions:\n\n"
        
        for i, match in enumerate(matches, 1):
            context += f"{i}. {match.metadata['text']}\n"
            context += f"   - From: {match.metadata['author']}\n"
            context += f"   - Date: {match.metadata['timestamp']}\n\n"
            
        return context
    
    def generate_openai_response(self, question: str, context: str, model: str = "gpt-4o-mini") -> str:
        """Generate response using OpenAI's GPT-4"""
        if model == "auto":
            model = "gpt-4o-mini"
        try:
            logger.info(f"Generating response using {model}")
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about Solana blockchain."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            answer = response.choices[0].message.content
            logger.info("Successfully generated response using GPT-4")
            return answer
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise
    
    def generate_sentence_transformers_response(self, question: str, context: str) -> str:
        """Generate response using sentence-transformers"""
        try:
            logger.info("Generating response using sentence-transformers")
            # Combine context and question
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
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
    
    def generate_response(self, question: str, model: str = "auto") -> str:
        """Generate response using the best available model"""
        try:
            # Get relevant context from Pinecone
            context = self.text_processor.get_relevant_context(question)
            
            return self.generate_openai_response(question, context, model)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def process_and_store_messages(self, messages: List[Any]):
        """Process messages and store them in the vector database"""
        for message in tqdm(messages, desc="Processing messages"):
            # Convert SQLAlchemy Message to dictionary if needed
            if not isinstance(message, dict):
                message_dict = {
                    'id': message.id,
                    'content': message.content,
                    'author': {
                        'id': message.author_id,
                        'username': message.author.username if message.author else 'Unknown'
                    },
                    'timestamp': message.timestamp,
                    'channel_id': message.channel_id
                }
                message = message_dict

            chunks = self.text_processor.process_message(message)
            self.text_processor.upsert_to_pinecone(chunks)
    
    def generate_response_with_context(self, question: str, model: str = "auto", top_k: int = 3):
        """
        Generate response and return both the answer and the top_k relevant context matches.
        """
        try:
            matches = self.text_processor.index.query(
                vector=self.text_processor.model.encode(question).tolist(),
                top_k=top_k,
                include_metadata=True
            ).matches

            context = self.format_context(matches)
            if model == "sentence-transformers":
                answer = self.generate_sentence_transformers_response(question, context)
            else:
                answer = self.generate_openai_response(question, context, model)
            return answer, matches
        except Exception as e:
            logger.error(f"Error generating response with context: {str(e)}")
            raise 