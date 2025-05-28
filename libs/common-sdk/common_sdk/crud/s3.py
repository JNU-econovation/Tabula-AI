import boto3

from botocore.client import Config as S3Config
from typing import Optional, Dict, Any

from common_sdk.config import settings
from common_sdk.exceptions import ExternalConnectionError, UploadFailedError
from common_sdk.get_logger import get_logger

# 로거 설정
logger = get_logger()

# S3 클래스
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
    def check_connection(self) -> bool:
        try:
            # 버킷 존재 여부 확인
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"[check_connection] S3 connection successful - Bucket: {self.bucket}")
            return True
        
        except Exception as e:
            logger.error(f"[check_connection] S3 connection failed: {str(e)}")
            raise ExternalConnectionError()
    
    # 파일 업로드(S3)
    def upload_note_file(self, file_path: str, user_id: int, space_id: str) -> Dict[str, Any]:
        """
        Args:
            file_path (str): 업로드할 파일 경로(pdf)
            user_id (int): 사용자 ID
            space_id (str): 학습 공간 ID
        """
        try:
            # S3 경로 생성
            s3_key = f"pdf/{user_id}/{space_id}.pdf"
            
            # 파일 업로드
            self.client.upload_file(
                file_path,
                self.bucket,
                s3_key,
                ExtraArgs={'ContentType': 'application/pdf'}
            )
            
            # S3 URL 생성
            s3_url = f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            
            logger.info(f"[upload_note_file] User: {user_id} - Upload successfully: {s3_key}")
            
            return s3_url
            
        except Exception as e:
            logger.error(f"[upload_note_file] User: {user_id} - Failed to upload: {str(e)}")
            raise UploadFailedError()
        
    # 원본 이미지 업로드 (origin)
    def upload_origin_image(self, file_path: str, user_id: int, space_id: str, result_id: str, page: int) -> Dict[str, Any]:
        """
        Args:
            file_path (str): 업로드할 이미지 파일 경로(png)
            user_id (int): 사용자 ID
            space_id (str): 학습 공간 ID
            result_id (str): 결과 ID
            page (int): 페이지 번호
            
        Returns:
            Dict[str, Any]: {
                "s3_key": str,  # S3에 저장된 파일의 키
                "bucket": str,  # S3 버킷 이름
                "url": str      # S3 파일 URL
            }
        """
        try:
            # S3 경로 생성: image/origin/{userId}/{spaceId}/{resultId}/{id}.png
            s3_key = f"image/origin/{user_id}/{space_id}/{result_id}/{page}.png"
            
            # 파일 업로드
            self.client.upload_file(
                file_path,
                self.bucket,
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # S3 URL 생성
            s3_url = f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            
            logger.info(f"User: {user_id} Origin image uploaded successfully to S3 - {s3_key}")
            
            return {
                "s3_key": s3_key,
                "bucket": self.bucket,
                "url": s3_url
            }
            
        except Exception as e:
            logger.error(f"User: {user_id} Origin image upload failed to S3 - {str(e)}")
            raise UploadFailedError()
        
    # 하이라이트 이미지 업로드 (post)
    def upload_post_image(self, file_path: str, user_id: int, space_id: str, result_id: str, page: int) -> Dict[str, Any]:
        """
        Args:
            file_path (str): 업로드할 하이라이트 이미지 파일 경로(png)
            user_id (int): 사용자 ID
            space_id (str): 학습 공간 ID
            result_id (str): 결과 ID
            page (int): 페이지 번호
            
        Returns:
            Dict[str, Any]: {
                "s3_key": str,  # S3에 저장된 파일의 키
                "bucket": str,  # S3 버킷 이름
                "url": str      # S3 파일 URL
            }
        """
        try:
            # S3 경로 생성: image/post/{userId}/{spaceId}/{resultId}/{id}.png
            s3_key = f"images/post/{user_id}/{space_id}/{result_id}/{page}.png"
            
            # 파일 업로드
            self.client.upload_file(
                file_path,
                self.bucket,
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # S3 URL 생성
            s3_url = f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            
            logger.info(f"User: {user_id} Post image uploaded successfully to S3 - {s3_key}")
            
            return {
                "s3_key": s3_key,
                "bucket": self.bucket,
                "url": s3_url
            }
            
        except Exception as e:
            logger.error(f"User: {user_id} Post image upload failed to S3 - {str(e)}")
            raise UploadFailedError()