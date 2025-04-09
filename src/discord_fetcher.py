import discord
from discord.ext import commands
import asyncio
from datetime import datetime
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from ..database.models import Message, User, Reaction
from ..database.connection import SessionLocal

load_dotenv()

class DiscordFetcher(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
    async def fetch_channel_messages(self, channel_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch messages from a specific channel"""
        channel = self.get_channel(channel_id)
        if not channel:
            raise ValueError(f"Channel {channel_id} not found")
            
        messages = []
        async for message in channel.history(limit=limit):
            messages.append({
                'id': str(message.id),
                'channel_id': str(message.channel.id),
                'content': message.content,
                'timestamp': message.created_at,
                'edited_timestamp': message.edited_at,
                'author': {
                    'id': str(message.author.id),
                    'username': message.author.name,
                    'global_name': message.author.global_name,
                    'avatar': message.author.avatar.key if message.author.avatar else None,
                    'discriminator': message.author.discriminator
                },
                'thread_id': str(message.thread.id) if message.thread else None,
                'reference_message_id': str(message.reference.message_id) if message.reference else None,
                'is_pinned': message.pinned,
                'has_embeds': bool(message.embeds),
                'has_attachments': bool(message.attachments),
                'reactions': [
                    {
                        'emoji_name': reaction.emoji.name,
                        'emoji_id': str(reaction.emoji.id) if reaction.emoji.id else None,
                        'count': reaction.count
                    }
                    for reaction in message.reactions
                ]
            })
        return messages

    async def save_messages_to_db(self, messages: List[Dict[str, Any]]):
        """Save fetched messages to the database"""
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
                        global_name=user_data['global_name'],
                        avatar=user_data['avatar'],
                        discriminator=user_data['discriminator']
                    )
                    db.add(user)
                
                # Create message
                message = Message(
                    id=msg_data['id'],
                    channel_id=msg_data['channel_id'],
                    content=msg_data['content'],
                    timestamp=msg_data['timestamp'],
                    edited_timestamp=msg_data['edited_timestamp'],
                    author_id=user_data['id'],
                    thread_id=msg_data['thread_id'],
                    reference_message_id=msg_data['reference_message_id'],
                    is_pinned=msg_data['is_pinned'],
                    has_embeds=msg_data['has_embeds'],
                    has_attachments=msg_data['has_attachments']
                )
                db.add(message)
                
                # Add reactions
                for reaction_data in msg_data['reactions']:
                    reaction = Reaction(
                        message_id=msg_data['id'],
                        emoji_name=reaction_data['emoji_name'],
                        emoji_id=reaction_data['emoji_id'],
                        count=reaction_data['count']
                    )
                    db.add(reaction)
            
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

async def main():
    bot = DiscordFetcher()
    await bot.start(os.getenv('DISCORD_TOKEN'))
    
    # Fetch messages from the Solana channel
    channel_id = int(os.getenv('SOLANA_CHANNEL_ID'))
    messages = await bot.fetch_channel_messages(channel_id)
    
    # Save messages to database
    await bot.save_messages_to_db(messages)
    
    await bot.close()

if __name__ == "__main__":
    asyncio.run(main()) 