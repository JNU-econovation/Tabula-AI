import pytz
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId

from common_sdk.config import settings
from common_sdk.get_logger import get_logger
from common_sdk.exceptions import ExternalConnectionError, UploadFailedError

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
                # logger.info(f"Connecting to MongoDB: {settings.MONGO_URI}")
                # logger.info(f"Database name: {settings.MONGO_DATABASE}")
                
                self.client = AsyncIOMotorClient(settings.MONGO_URI)
                self.sync_client = MongoClient(settings.MONGO_URI)
                self.db = self.client[settings.MONGO_DATABASE]
                
            except Exception as e:
                logger.error(f"[MongoDB] MongoDB connection failed: {str(e)}")
                raise ExternalConnectionError()
    
    # MongoDB 연결 상태 확인
    def check_connection(self) -> bool:
        try:
            # ping check
            self.db.command('ping')
            logger.info("[check_connection] MongoDB connection successful")
            return True
        
        except Exception as e:
            logger.error(f"[check_connection] MongoDB connection failed: {str(e)}")
            raise ExternalConnectionError()
    
    # 학습 공간 생성
    def create_space(self, 
                    user_id: str,
                    space_id: str,
                    folder_id: str,
                    file_url: str,
                    file_name: str,
                    lang_type: str,
                    file_domain: str,
                    space_name: str = None,
                    keywords: list = None) -> Dict[str, Any]:
        try:
            # ObjectId 생성
            object_id = ObjectId(space_id)
            
            # 현재 시간을 한국 시간으로 변환
            now = datetime.now(kst)
            
            space_data = {
                "_id": object_id,
                "folder_id": folder_id,
                "space_name": space_name,
                "file_url": file_url,
                "file_name": file_name,
                "lang_type": lang_type,
                "file_domain": file_domain,
                "keyword": keywords,
                "created_at": now,
                "updated_at": now,
                "is_deleted": False
            }
            
            # spaces 컬렉션에 저장 (동기 클라이언트 사용)
            self.sync_client[settings.MONGO_DATABASE].spaces.insert_one(space_data)
            logger.info(f"[create_space] User: {user_id} - Upload successfully: {object_id}")
            return space_data
            
        except Exception as e:
            logger.error(f"[create_space] User: {user_id} - Failed to upload: {str(e)}")
            raise UploadFailedError()