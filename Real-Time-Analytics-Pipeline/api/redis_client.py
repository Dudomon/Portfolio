"""Redis client for caching"""

import redis
from config import settings


class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    
    def get(self, key):
        """Get value by key"""
        return self.client.get(key)
    
    def set(self, key, value):
        """Set key-value pair"""
        return self.client.set(key, value)
    
    def setex(self, key, seconds, value):
        """Set key-value with expiration"""
        return self.client.setex(key, seconds, value)
    
    def delete(self, key):
        """Delete key"""
        return self.client.delete(key)
    
    def ping(self):
        """Ping Redis"""
        return self.client.ping()
