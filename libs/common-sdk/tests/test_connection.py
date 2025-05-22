import pytest
import os
import tempfile
from common_sdk.conn import MongoDB, S3Storage
from common_sdk.config import settings

@pytest.fixture
def mongodb():
    """MongoDB 연결을 위한 fixture"""
    db = MongoDB()
    yield db
    # 테스트 후 연결 종료
    db.close()

@pytest.fixture
def s3():
    """S3 연결을 위한 fixture"""
    return S3Storage()

@pytest.mark.asyncio
async def test_mongodb_connection(mongodb):
    """MongoDB 연결 테스트"""
    # 연결 확인
    assert mongodb.client is not None
    assert mongodb.sync_client is not None
    assert mongodb.db is not None
    
    # 데이터베이스 이름 확인
    assert mongodb.db.name == settings.MONGO_DATABASE
    
    # 간단한 CRUD 테스트
    collection = mongodb.db.test_collection
    
    # 데이터 삽입
    test_doc = {"test": "data"}
    result = await collection.insert_one(test_doc)
    assert result.inserted_id is not None
    
    # 데이터 조회
    found_doc = await collection.find_one({"test": "data"})
    assert found_doc is not None
    assert found_doc["test"] == "data"
    
    # 데이터 삭제
    delete_result = await collection.delete_one({"test": "data"})
    assert delete_result.deleted_count == 1

def test_s3_connection(s3):
    """S3 연결 테스트"""
    # 연결 확인
    assert s3._client is not None
    
    # 테스트용 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"test content")
        temp_file_path = temp_file.name
    
    try:
        # 파일 업로드 테스트
        s3_key = "test/test_file.txt"
        assert s3.upload_file(temp_file_path, s3_key) is True
        
        # 다운로드 테스트
        download_path = temp_file_path + ".downloaded"
        assert s3.download_file(s3_key, download_path) is True
        
        # 다운로드한 파일 내용 확인
        with open(download_path, 'rb') as f:
            content = f.read()
            assert content == b"test content"
        
        # presigned URL 생성 테스트
        url = s3.get_presigned_url(s3_key)
        assert url is not None
        assert settings.S3_BUCKET in url
        
        # 파일 삭제 테스트
        assert s3.delete_file(s3_key) is True
        
    finally:
        # 임시 파일 정리
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(download_path):
            os.unlink(download_path)

def test_s3_fileobj_upload(s3):
    """S3 파일 객체 업로드 테스트"""
    # 테스트용 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"test content")
        temp_file_path = temp_file.name
    
    try:
        # 파일 객체로 업로드 테스트
        s3_key = "test/test_fileobj.txt"
        with open(temp_file_path, 'rb') as file_obj:
            assert s3.upload_fileobj(file_obj, s3_key) is True
        
        # 업로드된 파일 삭제
        assert s3.delete_file(s3_key) is True
        
    finally:
        # 임시 파일 정리
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path) 