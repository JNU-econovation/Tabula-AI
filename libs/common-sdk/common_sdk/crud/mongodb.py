import pytz

from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from datetime import datetime

from common_sdk.config import settings
from common_sdk.get_logger import get_logger
from common_sdk.exceptions import ExternalConnectionError

logger = get_logger()

# 시간대 설정
kst = pytz.timezone("Asia/Seoul")

# MongoDB 클래스
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
            try:
                self.client = AsyncIOMotorClient(settings.MONGO_URI)
                self.sync_client = MongoClient(settings.MONGO_URI)
                self.db = self.client[settings.MONGO_DATABASE]
                logger.info("MongoDB connection successful")
            except Exception as e:
                logger.error(f"MongoDB connection failed: {str(e)}")
                raise ExternalConnectionError()
    
    # MongoDB 연결 상태 확인
    def check_connection(self) -> bool:
        try:
            # ping check
            self.db.command('ping')
            logger.info("MongoDB connection successful")
            return True
        
        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            raise ExternalConnectionError()
    
    # 학습 공간 생성
    def create_space(self, 
                    space_id: str,
                    folder_id: str,
                    file_url: str,
                    file_name: str,
                    lang_type: str,
                    file_domain: str,
                    space_name: str = None,
                    keywords: list = None) -> Dict[str, Any]:
        try:
            space_data = {
                "_id": space_id,
                "folder_id": folder_id,
                "space_name": space_name,
                "file_url": file_url,
                "file_name": file_name,
                "lang_type": lang_type,
                "file_domain": file_domain,
                "keyword": keywords,
                "created_at": datetime.now(kst),
                "updated_at": datetime.now(kst),
                "is_deleted": True
            }
            
            result = self.db.spaces.insert_one(space_data)
            logger.info(f"Space created successfully: {space_id}")
            return space_data
            
        except Exception as e:
            logger.error(f"Failed to create space: {str(e)}")
            raise