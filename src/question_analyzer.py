from typing import List, Dict, Any, Tuple
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
from loguru import logger
from database.models import Message, QuestionGroup
from database.connection import SessionLocal
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import os

load_dotenv()

class QuestionAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.question_pattern = re.compile(r'^(what|how|why|when|where|who|is|are|can|could|would|will|do|does|did|should|shall|may|might)\s+', re.IGNORECASE)
        self.question_mark_pattern = re.compile(r'\?$')
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
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
        response = self.openai_client.chat.completions.create(
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
    
    def cluster_questions(self, questions: List[Dict[str, Any]], eps: float = 0.3) -> List[List[Dict[str, Any]]]:
        """Cluster similar questions together using DBSCAN"""
        if not questions:
            return []
            
        # Get embeddings for all questions
        texts = [q['text'] for q in questions]
        embeddings = self.model.encode(texts)
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine').fit(embeddings)
        
        # Group questions by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # -1 means noise in DBSCAN
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(questions[idx])
        
        return list(clusters.values())
    
    def process_and_store_questions(self, messages: List[Dict[str, Any]], time_window_days: int = 30, use_llm: bool = False):
        """Process messages, extract questions, cluster them, and store in database"""
        # Extract questions
        questions = self.extract_questions(messages)
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
            logger.error(f"Error processing questions: {str(e)}")
            raise e
        finally:
            db.close()
    
    def _are_questions_similar(self, q1: str, q2: str, threshold: float = 0.8) -> bool:
        """Check if two questions are semantically similar"""
        embeddings = self.model.encode([q1, q2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity > threshold 