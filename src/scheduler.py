import asyncio
from datetime import datetime, time
import aiohttp
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from .bot_manager import BotManager
from ..database.models import Message, User
from ..database.connection import SessionLocal

load_dotenv()

class MessageScheduler:
    def __init__(self):
        self.bot_manager = BotManager()
        self.api_url = os.getenv('MESSAGE_API_URL')
        self.schedule_time = time(hour=0, minute=0)  # Default to midnight
        
    async def fetch_messages_from_api(self) -> List[Dict[str, Any]]:
        """Fetch messages from the external API"""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.api_url) as response:
                if response.status != 200:
                    raise Exception(f"API request failed with status {response.status}")
                return await response.json()
                
    async def process_messages(self, messages: List[Dict[str, Any]]):
        """Process messages and store them in the database"""
        db = SessionLocal()
        try:
            for msg_data in messages:
                # Create or update user
                user_data = msg_data['author']
                user = db.query(User).filter(User.id == user_data['id']).first()
                if not user:
                    user = User(
                        id=user_data['id'],
                        username=user_data['username'],
                        global_name=user_data.get('global_name'),
                        avatar=user_data.get('avatar'),
                        discriminator=user_data.get('discriminator')
                    )
                    db.add(user)
                
                # Create message
                message = Message(
                    id=msg_data['id'],
                    channel_id=msg_data['channel_id'],
                    content=msg_data['content'],
                    timestamp=datetime.fromisoformat(msg_data['timestamp']),
                    edited_timestamp=datetime.fromisoformat(msg_data['edited_timestamp']) if msg_data.get('edited_timestamp') else None,
                    author_id=user_data['id'],
                    thread_id=msg_data.get('thread_id'),
                    reference_message_id=msg_data.get('reference_message_id'),
                    is_pinned=msg_data.get('is_pinned', False),
                    has_embeds=msg_data.get('has_embeds', False),
                    has_attachments=msg_data.get('has_attachments', False)
                )
                db.add(message)
                
                # Post to mirror channels
                await self.bot_manager.post_message_to_mirror(user_data['id'], message)
            
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
            
    async def run_daily_task(self):
        """Run the daily message processing task"""
        while True:
            now = datetime.now().time()
            if now.hour == self.schedule_time.hour and now.minute == self.schedule_time.minute:
                try:
                    messages = await this.fetch_messages_from_api()
                    await this.process_messages(messages)
                except Exception as e:
                    print(f"Error in daily task: {e}")
                    
            # Sleep for 1 minute before checking again
            await asyncio.sleep(60)
            
    def start(self):
        """Start the scheduler"""
        asyncio.create_task(this.run_daily_task())
        
    async def cleanup(self):
        """Cleanup resources"""
        await this.bot_manager.cleanup() 