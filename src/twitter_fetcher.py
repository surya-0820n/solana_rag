import tweepy
from typing import List, Dict, Any
import os
from loguru import logger
from datetime import datetime, timedelta

class TwitterFetcher:
    def __init__(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        if not all([self.api_key, self.api_secret, self.access_token, self.access_token_secret]):
            raise ValueError("Missing Twitter API credentials")
            
        # Initialize Twitter client
        self.client = tweepy.Client(
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret
        )
        logger.info("TwitterFetcher initialized successfully")
        
    def fetch_user_tweets(self, username: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch tweets from a specific user within the last N days
        
        Args:
            username (str): Twitter username to fetch tweets from
            days_back (int): Number of days to look back
            
        Returns:
            List[Dict[str, Any]]: List of processed tweets
        """
        try:
            # Get user ID from username
            user = self.client.get_user(username=username)
            if not user.data:
                raise ValueError(f"User {username} not found")
                
            user_id = user.data.id
            
            # Calculate date range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            # Fetch tweets
            tweets = []
            for tweet in tweepy.Paginator(
                self.client.get_users_tweets,
                user_id,
                max_results=100,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=['created_at', 'public_metrics', 'entities'],
                expansions=['author_id'],
                user_fields=['name', 'username', 'profile_image_url']
            ):
                if tweet.data:
                    for t in tweet.data:
                        # Get user info from includes
                        user_info = next(
                            (u for u in tweet.includes['users'] if u.id == t.author_id),
                            None
                        )
                        
                        # Process tweet
                        processed_tweet = {
                            'id': str(t.id),
                            'text': t.text,
                            'created_at': t.created_at,
                            'author': {
                                'id': str(user_info.id),
                                'name': user_info.name,
                                'username': user_info.username,
                                'profile_image_url': user_info.profile_image_url
                            },
                            'metrics': {
                                'retweet_count': t.public_metrics['retweet_count'],
                                'reply_count': t.public_metrics['reply_count'],
                                'like_count': t.public_metrics['like_count'],
                                'quote_count': t.public_metrics['quote_count']
                            },
                            'entities': {
                                'hashtags': [tag['tag'] for tag in t.entities.get('hashtags', [])],
                                'mentions': [mention['username'] for mention in t.entities.get('mentions', [])],
                                'urls': [url['expanded_url'] for url in t.entities.get('urls', [])]
                            }
                        }
                        tweets.append(processed_tweet)
            
            logger.info(f"Fetched {len(tweets)} tweets from @{username}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error fetching tweets: {str(e)}")
            raise 