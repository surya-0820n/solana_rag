import os
from dotenv import load_dotenv
from database.models import Message, User
from database.connection import SessionLocal
import discord
from typing import Dict, Optional
from loguru import logger

load_dotenv()

class BotManager:
    def __init__(self):
        self.bots: Dict[str, discord.Client] = {}
        logger.info("Initialized BotManager")
        
    def create_bot_for_user(self, user_id: str, bot_token: str):
        """Create a new bot instance for a user"""
        if user_id not in self.bots:
            logger.info(f"Creating new bot for user {user_id}")
            intents = discord.Intents.default()
            intents.message_content = True
            bot = discord.Client(intents=intents)
            bot.run(bot_token)
            self.bots[user_id] = bot
            logger.info(f"Bot created successfully for user {user_id}")
            
    def setup_mirror_channel(self, 
                           user_id: str,
                           source_channel_id: int,
                           source_guild_id: int,
                           target_guild_id: int,
                           category_id: Optional[int] = None) -> int:
        """Set up a mirror channel for a user"""
        logger.info(f"Setting up mirror channel for user {user_id}")
        bot = self.bots.get(user_id)
        if not bot:
            logger.error(f"No bot found for user {user_id}")
            raise Exception("Bot not found for user")
            
        # Get source channel
        source_channel = bot.get_channel(source_channel_id)
        if not source_channel:
            logger.error(f"Source channel {source_channel_id} not found")
            raise Exception("Source channel not found")
            
        # Get target guild
        target_guild = bot.get_guild(target_guild_id)
        if not target_guild:
            logger.error(f"Target guild {target_guild_id} not found")
            raise Exception("Target guild not found")
            
        # Create mirror channel
        logger.info(f"Creating mirror channel in guild {target_guild_id}")
        mirror_channel = target_guild.create_text_channel(
            name=f"mirror-{source_channel.name}",
            category=target_guild.get_channel(category_id) if category_id else None
        )
        
        logger.info(f"Mirror channel created successfully with ID {mirror_channel.id}")
        return mirror_channel.id
        
    def post_message_to_mirror(self, user_id: str, message: Message):
        """Post a message to all mirror channels for a user"""
        logger.debug(f"Posting message {message.id} to mirror channels for user {user_id}")
        bot = self.bots.get(user_id)
        if not bot:
            logger.warning(f"No bot found for user {user_id}")
            return
            
        db = SessionLocal()
        try:
            # Get all mirror channels for this user
            mirror_channels = db.query(MirrorChannel).filter(
                MirrorChannel.user_id == user_id,
                MirrorChannel.source_channel_id == message.channel_id
            ).all()
            
            for mirror_channel in mirror_channels:
                channel = bot.get_channel(mirror_channel.target_channel_id)
                if channel:
                    logger.debug(f"Posting to mirror channel {mirror_channel.target_channel_id}")
                    channel.send(message.content)
                else:
                    logger.warning(f"Mirror channel {mirror_channel.target_channel_id} not found")
        except Exception as e:
            logger.error(f"Error posting to mirror channels: {str(e)}")
        finally:
            db.close()
            
    def cleanup(self):
        """Cleanup all bot instances"""
        logger.info("Cleaning up bot instances")
        for bot in self.bots.values():
            try:
                bot.close()
                logger.info("Bot closed successfully")
            except Exception as e:
                logger.error(f"Error closing bot: {str(e)}")
        self.bots.clear()
        logger.info("All bots cleaned up") 