from common_sdk.conn.mongodb import MongoDB
from common_sdk.config import settings

# MongoDB 연결 테스트
def test_mongodb_connection():
    # MongoDB 인스턴스 생성
    mongodb = MongoDB()
    
    # 연결 상태 확인
    result = mongodb.check_connection()
    
    # 테스트 검증
    assert result["success"] is True, "MongoDB Connection Failed"
    assert result["response"]["status"] == "connected", "MongoDB Status is not 'connected'"
    assert result["error"] is None, "Error Occurred"
    
    print("MongoDB Connection Test Success")

def test_mongodb_database_name():
    # MongoDB 인스턴스 생성
    mongodb = MongoDB()
    
    # 데이터베이스 이름 확인
    assert mongodb.db.name == settings.MONGO_DATABASE, "MongoDB Database Name is not configured"
    
    print("MongoDB Database Name Test Success")

if __name__ == "__main__":
    test_mongodb_connection()
    test_mongodb_database_name() 