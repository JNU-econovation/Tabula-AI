from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from ..config import settings

class MongoDB:
    _instance: Optional['MongoDB'] = None
    _client: Optional[AsyncIOMotorClient] = None
    _sync_client: Optional[MongoClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._client:
            self._client = AsyncIOMotorClient(settings.MONGO_URI)
            self._sync_client = MongoClient(settings.MONGO_URI)
            self._db = self._client[settings.MONGO_DATABASE]
    
    @property
    def client(self) -> AsyncIOMotorClient:
        """비동기 MongoDB 클라이언트를 반환합니다."""
        return self._client
    
    @property
    def sync_client(self) -> MongoClient:
        """동기 MongoDB 클라이언트를 반환합니다."""
        return self._sync_client
    
    @property
    def db(self) -> AsyncIOMotorDatabase:
        """데이터베이스 인스턴스를 반환합니다."""
        return self._db
    
    async def close(self):
        """MongoDB 연결을 종료합니다."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None 


# # 사용 예시
# from common_sdk.conn import MongoDB

# # MongoDB 사용
# mongodb = MongoDB()
# db = mongodb.db
# collection = db.your_collection