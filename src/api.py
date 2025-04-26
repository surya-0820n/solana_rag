from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from src.rag_system import RAGSystem
from src.discord_fetcher import DiscordFetcher
from src.bot_manager import BotManager
from src.scheduler import MessageScheduler
import os
from dotenv import load_dotenv
import requests

load_dotenv()

app = FastAPI(title="Solana RAG API")
rag_system = RAGSystem()
bot_manager = BotManager()
scheduler = MessageScheduler()

class Question(BaseModel):
    text: str
    model: Optional[str] = "auto"  # "auto", "openai", or "sentence-transformers"
    provide_relevant_context: bool = False
    top_k: int = 3

class QuestionResponse(BaseModel):
    answer: str
    relevant_context: Optional[list] = None

class MirrorChannelRequest(BaseModel):
    user_id: str
    source_channel_id: int
    source_guild_id: int
    target_guild_id: int
    bot_token: str
    category_id: Optional[int] = None

@app.post("/ask", response_model=QuestionResponse)
def ask_question(question: Question):
    """Ask a question about Solana"""
    try:
        if question.provide_relevant_context:
            answer, context_matches = rag_system.generate_response_with_context(
                question.text, question.model, question.top_k
            )
            # Format context for API response (not as a string, but as a list of dicts)
            relevant_context = [
                {
                    "text": match.metadata.get("text", ""),
                    "author": match.metadata.get("author", "Unknown"),
                    "timestamp": match.metadata.get("timestamp", "Unknown"),
                    "score": match.score,
                }
                for match in context_matches
            ]
            return QuestionResponse(answer=answer, relevant_context=relevant_context)
        else:
            answer = rag_system.generate_response(question.text, question.model)
            return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-knowledge-base")
def update_knowledge_base():
    """Fetch messages from Postgres Database and update the pinecone knowledge base"""
    try:
        bot = DiscordFetcher()
        bot.start(os.getenv('DISCORD_TOKEN'))
        
        # channel_id = int(os.getenv('SOLANA_CHANNEL_ID'))
        # messages = bot.fetch_channel_messages(channel_id)  

        messages = scheduler.fetch_messages_from_database(limit=100)

        # Process and store messages in the pinecone database
        rag_system.process_and_store_messages(messages)
        
        bot.close()
        return {"status": "success", "message": f"Processed {len(messages)} messages"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup-mirror")
def setup_mirror_channel(request: MirrorChannelRequest):
    """Set up a mirror channel for a user"""
    try:
        # Create bot for user if it doesn't exist
        bot_manager.create_bot_for_user(request.user_id, request.bot_token)
        
        # Create mirror channel
        mirror_channel_id = bot_manager.setup_mirror_channel(
            request.user_id,
            request.source_channel_id,
            request.source_guild_id,
            request.target_guild_id,
            request.category_id
        )
        
        return {
            "status": "success",
            "message": "Mirror channel created successfully",
            "mirror_channel_id": mirror_channel_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-messages")
def process_messages():
    """Manually trigger message processing from the API"""
    try:
        messages = scheduler.fetch_messages_from_api(
            headers={"Authorization": os.getenv('MESSAGE_API_TOKEN')},
            params={"limit": 100}
        )
        scheduler.process_messages(messages)
        return {
            "status": "success",
            "message": f"Processed {len(messages)} messages"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-pinecone")
def check_pinecone():
    """Check Pinecone index statistics"""
    try:
        stats = rag_system.text_processor.check_index_stats()
        return {
            "status": "success",
            "index_name": os.getenv('PINECONE_INDEX_NAME'),
            "total_vectors": stats['total_vector_count'],
            "dimension": stats['dimension']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-pinecone-index")
def create_pinecone_index(dimension: Optional[int] = 384):
    """Create a new Pinecone index with the correct dimension"""
    try:
        rag_system.text_processor.create_index(dimension)
        return {
            "status": "success",
            "message": "Pinecone index created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Start the message scheduler on API startup"""
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on API shutdown"""
    await scheduler.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 