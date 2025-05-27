from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient

from common_sdk.config import settings
from common_sdk.get_logger import get_logger

logger = get_logger()

class MongoDB:
    instance: Optional['MongoDB'] = None
    client: Optional[AsyncIOMotorClient] = None
    sync_client: Optional[MongoClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    
    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        if not self.client:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.sync_client = MongoClient(settings.MONGO_URI)
            self.db = self.client[settings.MONGO_DATABASE]
    
    # MongoDB 연결 상태 확인
    def check_connection(self) -> Dict[str, Any]:
        try:
            # ping check
            self.db.command('ping')
            logger.info("MongoDB connection successful")
            return {
                "success": True,
                "response": {"status": "connected"},
                "error": None
            }
        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            return {
                "success": False,
                "response": None,
                "error": {
                    "code": "MONGODB_CONNECTION_ERROR",
                    "reason": str(e)
                }
            }
    
    # TODO: CRUD 로직 구현 예정
    """
    @property
    def client(self) -> AsyncIOMotorClient:
        pass
    
    @property
    def sync_client(self) -> MongoClient:
        pass
    
    @property
    def db(self) -> AsyncIOMotorDatabase:
        pass
    
    async def close(self):
        pass
    """


# # 사용 예시
# from common_sdk.conn import MongoDB

# # MongoDB 사용
# mongodb = MongoDB()
# db = mongodb.db
# collection = db.your_collection