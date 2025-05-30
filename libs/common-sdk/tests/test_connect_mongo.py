from common_sdk.crud.mongodb import MongoDB
from common_sdk.config import settings
from datetime import datetime
import pytz
from bson import ObjectId

# 시간대 설정
kst = pytz.timezone("Asia/Seoul")

# MongoDB 연결 테스트
def test_mongodb_connection():
    # MongoDB 인스턴스 생성
    mongodb = MongoDB()
    
    # 연결 상태 확인
    result = mongodb.check_connection()
    
    # 테스트 검증
    assert result is True, "MongoDB Connection Failed"
    print("MongoDB Connection Test Success")

def test_mongodb_database_name():
    # MongoDB 인스턴스 생성
    mongodb = MongoDB()
    
    # 데이터베이스 이름 확인
    assert mongodb.db.name == settings.MONGO_DATABASE, "MongoDB Database Name is not configured"
    print(f"MongoDB Database Name: {mongodb.db.name}")
    print("MongoDB Database Name Test Success")

def test_mongodb_insert():
    # MongoDB 인스턴스 생성
    mongodb = MongoDB()
    
    # 테스트용 ObjectId 생성
    test_id = ObjectId()
    
    # 테스트 데이터
    test_data = {
        "_id": test_id,
        "folder_id": "test_folder_001",
        "space_name": "Test Space",
        "file_url": "https://test.com/test.pdf",
        "file_name": "test.pdf",
        "lang_type": "ko",
        "file_domain": "test",
        "keyword": ["test1", "test2"],
        "created_at": datetime.now(kst),
        "updated_at": datetime.now(kst),
        "is_deleted": False
    }
    
    try:
        # 기존 테스트 데이터가 있다면 삭제
        mongodb.sync_client[settings.MONGO_DATABASE].spaces.delete_one({"_id": test_id})
        
        # 데이터 삽입
        result = mongodb.sync_client[settings.MONGO_DATABASE].spaces.insert_one(test_data)
        print(f"Inserted document ID: {result.inserted_id}")
        
        # 삽입된 데이터 확인
        inserted_data = mongodb.sync_client[settings.MONGO_DATABASE].spaces.find_one({"_id": test_id})
        assert inserted_data is not None, "Data was not inserted"
        print("Inserted data:", inserted_data)
        
        # 테스트 데이터 삭제
        # mongodb.sync_client[settings.MONGO_DATABASE].spaces.delete_one({"_id": test_id})
        # print("Test data deleted successfully")
        
        print("MongoDB Insert Test Success")
        
    except Exception as e:
        print(f"MongoDB Insert Test Failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("\n=== MongoDB Connection Test ===")
    test_mongodb_connection()
    
    print("\n=== MongoDB Database Name Test ===")
    test_mongodb_database_name()
    
    print("\n=== MongoDB Insert Test ===")
    test_mongodb_insert() 