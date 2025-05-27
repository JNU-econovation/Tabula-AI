import os
from dotenv import load_dotenv
from pathlib import Path

# 경로 설정: note-sdk
NOTE_SDK_ROOT = os.path.dirname(__file__)

# note-sdk의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

class Settings:

    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME_KOR_DEN_CONTENTS = os.getenv("INDEX_NAME_KOR_DEN_CONTENTS")
    INDEX_NAME_ENG_DEN_CONTENTS = os.getenv("INDEX_NAME_ENG_DEN_CONTENTS")
    INDEX_NAME_KOR_SPA_CONTENTS = os.getenv("INDEX_NAME_KOR_SPA_CONTENTS")
    INDEX_NAME_ENG_SPA_CONTENTS = os.getenv("INDEX_NAME_ENG_SPA_CONTENTS")

    # 기본 설정
    SPACE_DIR = os.path.join(NOTE_SDK_ROOT, "space")

    # 작업 디렉토리 구조
    @staticmethod
    def get_space_dir(space_id: str) -> Path:
        """사용자별 작업 디렉토리 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id

    @staticmethod
    def get_markdown_path(space_id: str, document_id: str) -> Path:
        """마크다운 파일 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "md" / f"{document_id}.md"

    @staticmethod
    def get_image_dir(space_id: str) -> Path:
        """이미지 디렉토리 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "images"

    @staticmethod
    def get_keyword_path(space_id: str, document_id: str) -> Path:
        """키워드 파일 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "keyword" / f"{document_id}.json"
    
    @staticmethod
    def get_origin_dir(space_id: str) -> Path:
        """원본 파일 디렉토리 경로 반환"""
        return Path(Settings.SPACE_DIR) / space_id / "origin"

    @staticmethod
    def get_temp_dir(space_id: str) -> Path:
        """임시 디렉토리 경로 반환"""
        temp_dir = Path(Settings.SPACE_DIR) / space_id / "temp"
        return temp_dir

    @staticmethod
    def ensure_user_directories(space_id: str) -> None:
        """사용자별 필요한 디렉토리 생성"""
        directories = [
            Path(Settings.SPACE_DIR) / space_id / "md",
            Path(Settings.SPACE_DIR) / space_id / "images",
            Path(Settings.SPACE_DIR) / space_id / "keyword",
            Path(Settings.SPACE_DIR) / space_id / "origin",
            Path(Settings.SPACE_DIR) / space_id / "temp"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Settings 인스턴스 생성
settings = Settings() 