from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(String, primary_key=True)  # Discord message ID
    channel_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    edited_timestamp = Column(DateTime, nullable=True)
    author_id = Column(String, ForeignKey('users.id'), nullable=False)
    thread_id = Column(String, nullable=True) 
    reference_message_id = Column(String, nullable=True)
    is_pinned = Column(Boolean, default=False)
    has_embeds = Column(Boolean, default=False)
    has_attachments = Column(Boolean, default=False)
    
    # Relationships
    author = relationship("User", back_populates="messages")
    reactions = relationship("Reaction", back_populates="message", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True)  # Discord user ID
    username = Column(String, nullable=False)
    global_name = Column(String, nullable=True)
    avatar = Column(String, nullable=True)
    discriminator = Column(String, nullable=True)
    
    # Relationships
    messages = relationship("Message", back_populates="author")

class Reaction(Base):
    __tablename__ = 'reactions'
    
    id = Column(Integer, primary_key=True)
    message_id = Column(String, ForeignKey('messages.id'), nullable=False)
    emoji_name = Column(String, nullable=False)
    emoji_id = Column(String, nullable=True)
    count = Column(Integer, default=0)
    
    # Relationships
    message = relationship("Message", back_populates="reactions")

class QuestionGroup(Base):
    __tablename__ = 'question_groups'
    
    id = Column(Integer, primary_key=True)
    representative_question = Column(String)
    count = Column(Integer, default=1)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    examples = Column(JSON)  # Store list of similar questions
    theme = Column(String, nullable=True)  # Common theme of the questions 