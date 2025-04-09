from typing import List, Dict, Any
import openai
from .text_processor import TextProcessor
import os
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.text_processor = TextProcessor()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def format_context(self, matches: List[Dict[str, Any]]) -> str:
        """Format retrieved context for the language model"""
        context = "Here are some relevant excerpts from Solana community discussions:\n\n"
        
        for i, match in enumerate(matches, 1):
            context += f"{i}. {match.metadata['text']}\n"
            context += f"   - From: {match.metadata['author']}\n"
            context += f"   - Date: {match.metadata['timestamp']}\n\n"
            
        return context
    
    def generate_response(self, query: str) -> str:
        """Generate a response using RAG"""
        # Retrieve relevant context
        matches = self.text_processor.search_similar(query)
        
        # Format context
        context = self.format_context(matches)
        
        # Create prompt for the language model
        prompt = f"""You are a helpful assistant with expertise in Solana blockchain technology. 
        Use the following context from Solana community discussions to answer the user's question.
        If the context doesn't contain enough information to fully answer the question, say so.
        Always cite the sources you use from the provided context.

        Context:
        {context}

        Question: {query}

        Answer:"""
        
        # Generate response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with expertise in Solana blockchain technology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def process_and_store_messages(self, messages: List[Dict[str, Any]]):
        """Process messages and store them in the vector database"""
        for message in messages:
            chunks = self.text_processor.process_message(message)
            self.text_processor.upsert_to_pinecone(chunks) 