import boto3
from botocore.client import Config as S3Config
from typing import Optional, BinaryIO
from ..config import settings

class S3Storage:
    _instance: Optional['S3Storage'] = None
    _client: Optional[boto3.client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._client:
            self._client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
                config=S3Config(signature_version='s3v4')
            )
    
    def upload_file(self, file_path: str, s3_key: str) -> bool:
        """로컬 파일을 S3에 업로드합니다."""
        try:
            self._client.upload_file(file_path, settings.S3_BUCKET, s3_key)
            return True
        except Exception as e:
            print(f"Error uploading file to S3: {e}")
            return False
    
    def upload_fileobj(self, file_obj: BinaryIO, s3_key: str) -> bool:
        """파일 객체를 S3에 업로드합니다."""
        try:
            self._client.upload_fileobj(file_obj, settings.S3_BUCKET, s3_key)
            return True
        except Exception as e:
            print(f"Error uploading file object to S3: {e}")
            return False
    
    def download_file(self, s3_key: str, file_path: str) -> bool:
        """S3에서 파일을 다운로드합니다."""
        try:
            self._client.download_file(settings.S3_BUCKET, s3_key, file_path)
            return True
        except Exception as e:
            print(f"Error downloading file from S3: {e}")
            return False
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """파일에 대한 presigned URL을 생성합니다."""
        try:
            return self._client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.S3_BUCKET,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None
    
    def delete_file(self, s3_key: str) -> bool:
        """S3에서 파일을 삭제합니다."""
        try:
            self._client.delete_object(
                Bucket=settings.S3_BUCKET,
                Key=s3_key
            )
            return True
        except Exception as e:
            print(f"Error deleting file from S3: {e}")
            return False 
        

# # 사용 예시
# from common_sdk.conn import S3Storage

# # S3 사용
# s3 = S3Storage()
# s3.upload_file("local_file.txt", "s3_key.txt")