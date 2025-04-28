from datetime import datetime, time
import requests
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from src.bot_manager import BotManager
from database.models import Message, User
from database.connection import SessionLocal
import time as time_module
from loguru import logger
from src.rag_system import RAGSystem
from src.question_analyzer import QuestionAnalyzer

load_dotenv()

class MessageScheduler:
    def __init__(self):
        self.bot_manager = BotManager()
        self.api_url = os.getenv('MESSAGE_API_URL')
        self.schedule_time = time(hour=0, minute=0)  # Default to midnight
        self.rag_system = RAGSystem()
        self.question_analyzer = QuestionAnalyzer()
        
    def fetch_messages_from_api(self, 
                              params: Optional[Dict[str, Any]] = None, 
                              headers: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Fetch messages from the external API
        
        Args:
            params (Dict[str, Any], optional): Query parameters to send with the request
            headers (Dict[str, str], optional): Headers to send with the request
            
        Returns:
            List[Dict[str, Any]]: List of messages from the API
        """
        logger.info(f"Calling API: {self.api_url} with params: {params} and headers: {headers}")
        response = requests.get(
            self.api_url,
            params=params or {},
            headers=headers or {}
        )
        
        if response.status_code != 200:
            logger.error(f"API request failed with status {response.status_code} and error {response.text}")
            raise Exception(f"API request failed with status {response.status_code}")
            
        logger.info("SUCCESS: Messages fetched successfully")
        return response.json()
    
    def process_messages(self, messages: List[Dict[str, Any]]):
        """Process messages and store them in the postgres database"""
        db = SessionLocal()
        try:
            # First, collect all unique users
            unique_users = {}
            for message in messages:
                author = message['author']
                user_id = author['id']
                if user_id not in unique_users:
                    unique_users[user_id] = author

            # Add unique users first
            for user_id, author in unique_users.items():
                existing_user = db.query(User).filter(User.id == user_id).first()
                if not existing_user:
                    logger.info(f"Adding new user to database: {user_id}")
                    user = User(
                        id=user_id,
                        username=author['username'],
                        global_name=author.get('global_name'),
                        avatar=author.get('avatar'),
                        discriminator=author.get('discriminator')
                    )
                    db.add(user)
                    logger.info(f"User {user_id} added to database")
                else:
                    logger.debug(f"User {user_id} already exists in database")

            # Commit users first to avoid unique constraint violations
            db.commit()

            # Now process messages
            for message in messages:
                # Check if message already exists
                existing_message = db.query(Message).filter(Message.id == message['id']).first()
                if existing_message:
                    logger.debug(f"Message {message['id']} already exists, skipping")
                    continue

                # Create message
                logger.info(f"Adding new message to database: {message['id']}")
                new_message = Message(
                    id=message['id'],
                    channel_id=message['channel_id'],
                    content=message['content'],
                    timestamp=datetime.fromisoformat(message['timestamp']),
                    edited_timestamp=datetime.fromisoformat(message['edited_timestamp']) if message.get('edited_timestamp') else None,
                    author_id=message['author']['id'],
                    thread_id=message.get('thread_id'),
                    reference_message_id=message.get('reference_message_id'),
                    is_pinned=message.get('is_pinned', False),
                    has_embeds=message.get('has_embeds', False),
                    has_attachments=message.get('has_attachments', False)
                )
                db.add(new_message)

            db.commit()
            logger.info(f"Successfully processed {len(messages)} messages")
            
            # Store messages in Pinecone
            self.rag_system.process_and_store_messages(messages)
            
            # Process questions from new messages
            self.question_analyzer.process_and_store_questions(messages)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error processing messages: {str(e)}")
            raise e
        finally:
            db.close()

    def fetch_messages_from_database(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch messages from the postgres database"""
        db = SessionLocal()
        try:
            # Fetch messages with author relationship
            messages = db.query(Message).join(User).order_by(Message.timestamp.desc()).limit(limit).all()
            
            # Convert to dictionaries while session is still open
            message_dicts = []
            for message in messages:
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
                message_dicts.append(message_dict)
            
            return message_dicts
        finally:
            db.close()

    def run_scheduled_task(self):
        """Run the scheduled task to fetch and process messages"""
        try:
            logger.info("Running scheduled message processing")
            messages = self.fetch_messages_from_api()
            self.process_messages(messages)
            return len(messages)
        except Exception as e:
            logger.error(f"Error in scheduled task: {e}")
            raise e

    def run_daily_task(self):
        """Run the daily message processing task"""
        logger.info("Starting daily task scheduler")
        while True:
            now = datetime.now().time()
            if now.hour == self.schedule_time.hour and now.minute == self.schedule_time.minute:
                try:
                    self.run_scheduled_task()
                except Exception as e:
                    logger.error(f"Error in daily task: {e}")
                    
            # Sleep for 1 minute before checking again
            time_module.sleep(60)
            
    def start(self):
        """Start the scheduler"""
        logger.info("Starting message scheduler")
        import threading
        thread = threading.Thread(target=self.run_daily_task)
        thread.daemon = True
        thread.start()
        
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up scheduler resources")
        self.bot_manager.cleanup() 