import boto3

from botocore.client import Config as S3Config
from typing import Optional, Dict, Any

from common_sdk.config import settings
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()

class S3Storage:
    instance: Optional['S3Storage'] = None
    client: Optional[boto3.client] = None
    bucket: str = settings.S3_BUCKET
    
    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance
    
    def __init__(self):
        if not self.client:
            self.client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
                config=S3Config(signature_version='s3v4')
            )
            
    # S3 연결 상태 확인
    def check_connection(self) -> Dict[str, Any]:
        try:
            # 버킷 존재 여부 확인
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3 connection successful - Bucket: {self.bucket}")
            return {
                "success": True,
                "response": {
                    "status": "connected",
                    "bucket": self.bucket
                },
                "error": None
            }
        except Exception as e:
            logger.error(f"S3 connection failed: {str(e)}")
            return {
                "success": False,
                "response": None,
                "error": {
                    "code": "S3_CONNECTION_ERROR",
                    "reason": str(e)
                }
            }
    
    # TODO: CRUD 로직 구현 예정
    """
    def upload_file(self, file_path: str, s3_key: str) -> bool:
        pass
    
    def upload_fileobj(self, file_obj: BinaryIO, s3_key: str) -> bool:
        pass
    
    def download_file(self, s3_key: str, file_path: str) -> bool:
        pass
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        pass
    
    def delete_file(self, s3_key: str) -> bool:
        pass
    """

# # 사용 예시
# from common_sdk.conn import S3Storage

# # S3 사용
# s3 = S3Storage()
# s3.upload_file("local_file.txt", "s3_key.txt")