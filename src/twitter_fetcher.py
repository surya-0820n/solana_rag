import os
import requests
from loguru import logger
from datetime import datetime, timedelta

class TwitterFetcher:
    def __init__(self):
        self.bearer_token = os.getenv('X_BEARER_TOKEN')
        if not self.bearer_token:
            raise ValueError("Missing X_BEARER_TOKEN in environment")
        self.headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        logger.info("TwitterFetcher initialized with Bearer Token")

    def fetch_user_tweets(self, username: str, days_back: int = 7):
        # 1. Get user ID
        user_url = f"https://api.twitter.com/2/users/by/username/{username}"
        user_resp = requests.get(user_url, headers=self.headers)
        user_resp.raise_for_status()
        user_id = user_resp.json()['data']['id']

        # 2. Get tweets
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        start_time_str = start_time.replace(microsecond=0).isoformat() + 'Z'
        end_time_str = end_time.replace(microsecond=0).isoformat() + 'Z'
        tweets_url = (
            f"https://api.twitter.com/2/users/{user_id}/tweets"
            f"?max_results=100"
            f"&start_time={start_time_str}"
            f"&end_time={end_time_str}"
            f"&tweet.fields=created_at,public_metrics,entities"
        )
        tweets_resp = requests.get(tweets_url, headers=self.headers)
        tweets_resp.raise_for_status()
        tweets = tweets_resp.json().get('data', [])
        logger.info(f"Fetched {len(tweets)} tweets from @{username}")
        return tweets 
    



if __name__ == "__main__":
    fetcher = TwitterFetcher()
    tweets = fetcher.fetch_user_tweets("solana", 30)
    print(tweets)