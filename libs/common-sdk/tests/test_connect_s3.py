from common_sdk.conn.s3 import S3Storage
from common_sdk.config import settings

# S3 연결 상태 테스트
def test_s3_connection():

    # S3 인스턴스 생성
    s3 = S3Storage()
    
    # 연결 상태 확인
    result = s3.check_connection()
    
    # 테스트 검증
    assert result["success"] is True, "S3 Connection Failed"
    assert result["response"]["status"] == "connected", "S3 Status is not 'connected'"
    assert result["response"]["bucket"] == settings.S3_BUCKET, "S3 Bucket is not configured"
    assert result["error"] is None, "Error Occurred"
    
    print("S3 Connection Test Success")

if __name__ == "__main__":
    test_s3_connection()
