import discord
from discord.ext import commands
import asyncio
from typing import Dict, Optional
import os
from dotenv import load_dotenv
from ..database.models import Message, User
from ..database.connection import SessionLocal

load_dotenv()

class BotManager:
    def __init__(self):
        self.bots: Dict[str, commands.Bot] = {}
        self.mirror_channels: Dict[str, Dict[str, int]] = {}  # {user_id: {source_channel_id: mirror_channel_id}}
        
    async def create_bot_for_user(self, user_id: str, token: str) -> commands.Bot:
        """Create a new bot for a user if it doesn't exist"""
        if user_id in self.bots:
            return self.bots[user_id]
            
        intents = discord.Intents.default()
        intents.message_content = True
        bot = commands.Bot(command_prefix='!', intents=intents)
        
        @bot.event
        async def on_ready():
            print(f'Bot {bot.user} is ready!')
            
        await bot.start(token)
        self.bots[user_id] = bot
        return bot
        
    async def setup_mirror_channel(
        self,
        user_id: str,
        source_channel_id: int,
        source_guild_id: int,
        target_guild_id: int,
        category_id: Optional[int] = None
    ) -> int:
        """Create a mirror channel for a source channel"""
        bot = self.bots.get(user_id)
        if not bot:
            raise ValueError(f"No bot found for user {user_id}")
            
        # Get source channel info
        source_guild = bot.get_guild(source_guild_id)
        if not source_guild:
            raise ValueError(f"Source guild {source_guild_id} not found")
            
        source_channel = source_guild.get_channel(source_channel_id)
        if not source_channel:
            raise ValueError(f"Source channel {source_channel_id} not found")
            
        # Get target guild
        target_guild = bot.get_guild(target_guild_id)
        if not target_guild:
            raise ValueError(f"Target guild {target_guild_id} not found")
            
        # Get category if specified
        category = None
        if category_id:
            category = target_guild.get_channel(category_id)
            if not category:
                raise ValueError(f"Category {category_id} not found")
        
        # Create mirror channel
        mirror_channel = await target_guild.create_text_channel(
            name=f"mirror-{source_channel.name}",
            topic=f"Mirror of {source_channel.name} from {source_guild.name}",
            category=category
        )
        
        # Store mapping
        if user_id not in self.mirror_channels:
            self.mirror_channels[user_id] = {}
        self.mirror_channels[user_id][str(source_channel_id)] = mirror_channel.id
        
        return mirror_channel.id
        
    async def post_message_to_mirror(self, user_id: str, message: Message):
        """Post a message to its mirror channel"""
        if user_id not in self.mirror_channels:
            return
            
        source_channel_id = str(message.channel_id)
        if source_channel_id not in self.mirror_channels[user_id]:
            return
            
        bot = self.bots.get(user_id)
        if not bot:
            return
            
        mirror_channel_id = self.mirror_channels[user_id][source_channel_id]
        mirror_channel = bot.get_channel(mirror_channel_id)
        if not mirror_channel:
            return
            
        # Create embed for the message
        embed = discord.Embed(
            description=message.content,
            timestamp=message.timestamp,
            color=discord.Color.blue()
        )
        
        # Add author information
        db = SessionLocal()
        try:
            author = db.query(User).filter(User.id == message.author_id).first()
            if author:
                embed.set_author(
                    name=author.global_name or author.username,
                    icon_url=author.avatar if author.avatar else None
                )
        finally:
            db.close()
            
        await mirror_channel.send(embed=embed)
        
    async def cleanup(self):
        """Cleanup all bots"""
        for bot in self.bots.values():
            await bot.close()
        self.bots.clear()
        self.mirror_channels.clear() 