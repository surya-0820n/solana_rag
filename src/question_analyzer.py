from typing import List, Dict, Any, Tuple
import re
from sklearn.cluster import DBSCAN
import numpy as np
from loguru import logger
from database.models import Message, QuestionGroup, Base
from database.connection import SessionLocal, engine
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import os
from src.utils.singletons import get_model
from sqlalchemy import text

load_dotenv()

class QuestionAnalyzer:
    def __init__(self):
        """Initialize the question analyzer"""
        self.model = get_model()
        self.question_pattern = re.compile(r'^(what|how|why|when|where|who|is|are|can|could|would|will|do|does|did|should|shall|may|might)\s+', re.IGNORECASE)
        self.question_mark_pattern = re.compile(r'\?$')
        # Initialize OpenAI client with API key
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Ensure database tables exist
        self._ensure_tables_exist()
        logger.info("QuestionAnalyzer initialized successfully")
        
    def _ensure_tables_exist(self):
        """Ensure all required database tables exist"""
        try:
            # Create all tables (including QuestionGroup, Message, User, etc.)
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating/verifying database tables: {str(e)}")
            raise
        
    def is_question(self, text: str) -> bool:
        """Check if a text is a question"""
        # Remove URLs and code blocks
        text = re.sub(r'http\S+|```[\s\S]*?```|`[^`]*`', '', text)
        text = text.strip()
        
        # Check if it ends with a question mark or starts with a question word
        return bool(self.question_mark_pattern.search(text)) or bool(self.question_pattern.match(text))
        
    def extract_questions(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract questions from messages"""
        questions = []
        for message in messages:
            if self.is_question(message['content']):
                questions.append({
                    'id': message['id'],
                    'text': message['content'],
                    'author': message['author']['username'],
                    'timestamp': message['timestamp'],
                    'channel_id': message['channel_id']
                })
        return questions
        
    def cluster_questions_with_llm(self, questions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster similar questions together using LLM"""
        logger.info(f"Clustering {len(questions)} questions with LLM")
        if not questions:
            return []
            
        # Prepare the prompt
        questions_text = "\n".join([f"{i+1}. {q['text']}" for i, q in enumerate(questions)])
        prompt = f"""Please group these questions into clusters based on their meaning. 
 Questions that ask about the same thing should be in the same group, even if they use different words.

 Questions:
 {questions_text}

 Please respond in the following format:
 Group 1:
 - Question numbers: 1, 3, 5
 - Common theme: [brief description of what these questions are asking]

 Group 2:
 - Question numbers: 2, 4
 - Common theme: [brief description of what these questions are asking]

 ...and so on for all groups."""

        # Get LLM response
        response = self.openai_client.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that groups similar questions together."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for more deterministic results
        )
        
        # Parse the response and create clusters
        clusters = []
        current_group = []
        current_theme = ""
        
        for line in response.choices[0].message.content.split('\n'):
            if line.startswith('Group'):
                if current_group:
                    clusters.append({
                        'questions': current_group,
                        'theme': current_theme
                    })
                current_group = []
                current_theme = ""
            elif line.startswith('- Question numbers:'):
                numbers = [int(n.strip()) for n in line.split(':')[1].split(',')]
                current_group = [questions[n-1] for n in numbers]
            elif line.startswith('- Common theme:'):
                current_theme = line.split(':')[1].strip()
                
        if current_group:
            clusters.append({
                'questions': current_group,
                'theme': current_theme
            })
            
        return clusters
    
    def cluster_questions(self, questions: List[Dict[str, Any]], method: str = 'dbscan') -> List[Dict[str, Any]]:
        """Cluster similar questions together"""
        if method == 'dbscan':
            return self._cluster_with_dbscan(questions)
        else:
            return self._cluster_with_llm(questions)
            
    def _cluster_with_dbscan(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Clustering {len(questions)} questions with DBSCAN")
        from pprint import pprint
        pprint(questions)
        """Cluster questions using DBSCAN"""
        # Create embeddings for questions
        embeddings = self.openai_client.embeddings.create(
            input=[q['text'] for q in questions],
            model="text-embedding-ada-002"
        ).data
        
        # Convert to numpy array
        X = np.array([e.embedding for e in embeddings])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
        
        # Group questions by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(questions[i])
        
        
        # Convert to list of groups
        groups = []
        for label, questions in clusters.items():
            if label != -1:  # Skip noise points
                groups.append({
                    'questions': questions,
                    'count': len(questions),
                    'theme': self._extract_theme(questions)
                })
        logger.info(f"DBSCAN successfully clustered {len(groups)} groups of questions")
        return groups
        
    def _cluster_with_llm(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Clustering {len(questions)} questions with LLM")
        """Cluster questions using LLM"""
        try:
            # Prepare the prompt
            questions_text = "\n".join([f"- {q['text']}" for q in questions])
            prompt = f"""
            Analyze the following questions and group them by theme. 
            Return the groups in the format: "Theme: [theme name]\nQuestions: [comma-separated question numbers]"
            
            Questions:
            {questions_text}
            """
            
            # Get response from OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes and groups questions by theme."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Process the response
            answer = response.choices[0].message.content
            logger.info("Successfully clustered questions using LLM")
            
            # Parse the response and create groups
            groups = []
            current_theme = None
            current_questions = []
            
            for line in answer.split('\n'):
                if line.startswith('Theme:'):
                    if current_theme and current_questions:
                        groups.append({
                            'theme': current_theme,
                            'questions': current_questions
                        })
                    current_theme = line.split('Theme:')[1].strip()
                    current_questions = []
                elif line.startswith('Questions:'):
                    question_indices = [int(idx.strip()) for idx in line.split('Questions:')[1].split(',')]
                    current_questions = [questions[idx-1] for idx in question_indices]
            
            if current_theme and current_questions:
                groups.append({
                    'theme': current_theme,
                    'questions': current_questions
                })
            logger.info(f"LLM successfully clustered {len(groups)} groups of questions")
            return groups
            
        except Exception as e:
            logger.error(f"Error clustering questions with LLM: {str(e)}")
            raise
        
    def _extract_theme(self, questions: List[Dict[str, Any]]) -> str:
        """Extract a common theme from a group of questions"""
        if len(questions) == 1:
            return questions[0]['text']
            
        # Use LLM to extract common theme
        response = self.openai_client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts the common theme from a group of questions."},
                {"role": "user", "content": f"What is the common theme of these questions?\n\n" + "\n".join([f"- {q['text']}" for q in questions])}
            ]
        )
        
        return response.choices[0].message.content.strip()
        
    def process_and_store_questions(self, messages: List[Dict[str, Any]], time_window_days: int = 30, use_llm: bool = False):
        """Process messages, extract questions, cluster them, and store in database"""
        try:
            # Extract questions
            questions = self.extract_questions(messages)
            logger.info(f"Extracted {len(questions)} questions from postgreSQL database")
            if not questions:
                return []
                
            # Cluster similar questions
            if use_llm:
                clusters = self.cluster_questions_with_llm(questions)
            else:
                clusters = self.cluster_questions(questions)
            
            # Store in database
            db = SessionLocal()
            try:
                # Get existing question groups
                existing_groups = {group.representative_question: group for group in db.query(QuestionGroup).all()}
                
                # Process each cluster
                for cluster in clusters:
                    if use_llm:
                        questions = cluster['questions']
                        theme = cluster['theme']
                    else:
                        questions = cluster
                        theme = None
                    
                    # Sort by timestamp to get the earliest question as representative
                    questions.sort(key=lambda x: x['timestamp'])
                    representative = questions[0]
                    
                    # Check if similar group exists
                    existing_group = None
                    for group in existing_groups.values():
                        if self._are_questions_similar(representative['text'], group.representative_question):
                            existing_group = group
                            break
                    
                    if existing_group:
                        # Update existing group
                        existing_group.count += len(questions)
                        existing_group.last_seen = max(q['timestamp'] for q in questions)
                        existing_group.examples = existing_group.examples + [q['text'] for q in questions]
                    else:
                        # Create new group
                        new_group = QuestionGroup(
                            representative_question=representative['text'],
                            count=len(questions),
                            first_seen=representative['timestamp'],
                            last_seen=max(q['timestamp'] for q in questions),
                            examples=[q['text'] for q in questions],
                            theme=theme
                        )
                        db.add(new_group)
                
                db.commit()
                
                # Get top questions from the last time_window_days
                cutoff_date = datetime.now() - timedelta(days=time_window_days)
                top_questions = db.query(QuestionGroup)\
                    .filter(QuestionGroup.last_seen >= cutoff_date)\
                    .order_by(QuestionGroup.count.desc())\
                    .limit(10)\
                    .all()
                
                return [{
                    'question': group.representative_question,
                    'count': group.count,
                    'examples': group.examples,
                    'theme': group.theme
                } for group in top_questions]
                
            except Exception as e:
                db.rollback()
                logger.error(f"Database error in process_and_store_questions: {str(e)}")
                raise
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Error in process_and_store_questions: {str(e)}")
            raise
        
    def _are_questions_similar(self, q1: str, q2: str, threshold: float = 0.8) -> bool:
        """Check if two questions are semantically similar"""
        embeddings = self.model.encode([q1, q2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity > threshold
        
    def get_common_questions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get common questions from the last N days"""
        logger.info(f"Getting common questions from the last {days} days")
        db = SessionLocal()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            groups = db.query(QuestionGroup).filter(
                QuestionGroup.created_at >= cutoff_date
            ).order_by(
                QuestionGroup.question_count.desc()
            ).all()
            
            return [{
                'theme': group.theme,
                'count': group.question_count,
                'questions': group.questions,
                'authors': group.authors,
                'timestamps': group.timestamps,
                'channel_ids': group.channel_ids
            } for group in groups]
        finally:
            db.close() 