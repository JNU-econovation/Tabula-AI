import pytz
from typing import Optional, Dict, Any, List
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
    
    # 학습 공간 언어 타입 조회
    def get_space_lang_type(self, space_id: str) -> str:
        try:
            # ObjectId 생성
            object_id = ObjectId(space_id)
            
            # spaces 컬렉션에서 해당 space_id의 문서 조회
            space = self.sync_client[settings.MONGO_DATABASE].spaces.find_one(
                {"_id": object_id, "is_deleted": False},
                {"lang_type": 1}  # lang_type 필드만 반환 (MongoDB projection)
            )
            
            if space:
                logger.info(f"Language type found for space {space_id}: {space['lang_type']}")
                return space['lang_type']
            else:
                logger.warning(f"Space not found or deleted: {space_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get language type for space {space_id}: {str(e)}")
            raise

    # 학습 공간 키워드 조회
    async def get_space_keywords(self, space_id: str) -> Dict[str, Any]:
        """MongoDB에서 space_id를 통해 키워드 데이터 조회"""
        try:
            # space_id를 ObjectId로 변환
            object_id = ObjectId(space_id)
            
            # spaces 컬렉션에서 해당 document 조회
            collection = self.db.spaces
            document = await collection.find_one({"_id": object_id})
            
            if not document:
                logger.warning(f"No document found for space_id: {space_id}")
                return None
            
            # keyword 필드 추출
            keyword_data = document.get("keyword")
            
            if not keyword_data:
                logger.warning(f"keyword field is empty for space_id: {space_id}")
                return None
                
            # 딕셔너리 형태 처리 (dev 구조)
            if isinstance(keyword_data, dict):
                logger.info(f"Retrieved keyword data (dict) for space_id: {space_id} - {keyword_data.get('name', 'Unknown')}")
                return keyword_data
            
            # 기존 리스트 형태 처리 (하위 호환성)
            elif isinstance(keyword_data, list):
                if len(keyword_data) > 0:
                    logger.info(f"Retrieved keyword data (list) for space_id: {space_id} - converting first item")
                    return keyword_data[0]  # 첫 번째 요소 반환
                else:
                    logger.warning(f"keyword field is empty list for space_id: {space_id}")
                    return None
            
            # 예상치 못한 형태
            else:
                logger.warning(f"keyword field is unexpected type ({type(keyword_data)}) for space_id: {space_id}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to fetch keyword data from MongoDB: {e}")
            return None

    # 학습 결과물 저장
    def create_result(self,
                     space_id: str,
                     origin_result_url: list,
                     wrong_answers: list = None,
                     missing_answers: list = None) -> Dict[str, Any]:
        try:
            # ObjectId 생성 (새로운 결과 ID)
            result_id = ObjectId()
            
            # space_id를 ObjectId로 변환
            space_object_id = ObjectId(space_id)
            
            result_data = {
                "_id": result_id,
                "space_id": space_object_id,
                "origin_result_url": origin_result_url,
                "wrong_answers": wrong_answers,
                "missing_answers": missing_answers if missing_answers else [],
                "created_at": datetime.now(kst),
                "updated_at": datetime.now(kst),
                "is_deleted": False
            }
            
            # results 컬렉션에 저장 (동기 클라이언트 사용)
            self.sync_client[settings.MONGO_DATABASE].results.insert_one(result_data)
            logger.info(f"Result created successfully in results collection: {result_id}")
            
            # 저장된 데이터 반환 (ObjectId를 문자열로 변환)
            result_data['_id'] = str(result_id)
            result_data['space_id'] = str(space_object_id)
            result_data['created_at'] = result_data['created_at'].isoformat()
            result_data['updated_at'] = result_data['updated_at'].isoformat()
            
            return result_data
            
        except Exception as e:
            logger.error(f"Failed to create result: {str(e)}")
            raise