from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from .rag_system import RAGSystem
from .discord_fetcher import DiscordFetcher
from .bot_manager import BotManager
from .scheduler import MessageScheduler
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Solana RAG API")
rag_system = RAGSystem()
bot_manager = BotManager()
scheduler = MessageScheduler()

class Question(BaseModel):
    text: str

class QuestionResponse(BaseModel):
    answer: str

class MirrorChannelRequest(BaseModel):
    user_id: str
    source_channel_id: int
    source_guild_id: int  # Server where the source channel is
    target_guild_id: int  # Server where you want to create the mirror channel
    bot_token: str
    category_id: Optional[int] = None  # Optional category to create the channel in

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(question: Question):
    """Ask a question about Solana"""
    try:
        answer = rag_system.generate_response(question.text)
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fetch-messages")
async def fetch_messages():
    """Fetch new messages from Discord and update the knowledge base"""
    try:
        bot = DiscordFetcher()
        await bot.start(os.getenv('DISCORD_TOKEN'))
        
        channel_id = int(os.getenv('SOLANA_CHANNEL_ID'))
        messages = await bot.fetch_channel_messages(channel_id)
        
        # Process and store messages
        rag_system.process_and_store_messages(messages)
        
        await bot.close()
        return {"status": "success", "message": f"Processed {len(messages)} messages"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/setup-mirror")
async def setup_mirror_channel(request: MirrorChannelRequest):
    """Set up a mirror channel for a user"""
    try:
        # Create bot for user if it doesn't exist
        await bot_manager.create_bot_for_user(request.user_id, request.bot_token)
        
        # Create mirror channel
        mirror_channel_id = await bot_manager.setup_mirror_channel(
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
async def process_messages():
    """Manually trigger message processing from the API"""
    try:
        messages = await scheduler.fetch_messages_from_api()
        await scheduler.process_messages(messages)
        return {
            "status": "success",
            "message": f"Processed {len(messages)} messages"
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